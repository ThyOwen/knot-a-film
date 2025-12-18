//
//  GraphRender.swift
//  MetalTest
//
//  Created by Owen O'Malley on 8/18/25.
//

import MetalKit
import Foundation

struct ScreenTransform {
    var offset : SIMD2<Float32> = .init(x: 0, y: 0)
    var zoom : SIMD2<Float32> = .init(x: 0.5, y: 0.5)
}

struct GraphParams<N : Numeric & FloatingPoint> {
    
    struct TileParams {
        let tileSize : N
        let numTiles : Int
    }
    
    let widthParams : TileParams
    let heightParams : TileParams
    
    let edgeRepulsion : Double = 1.0
    let edgeAttraction : Double = 0.1
    let damping : Double = 0.001
    let epsilon : Double = 1e-03
    
    let numNodes : Int = 5000
        
    init(tileSize: N, numTiles: Int) {
        self.widthParams = .init(tileSize: tileSize, numTiles: numTiles)
        self.heightParams = .init(tileSize: tileSize, numTiles: numTiles)
    }
}

final class GraphRenderer : NSObject, MTKViewDelegate {

    var graph : Graph
    
    let metalDevice : MTLDevice
    var commandQueue: MTLCommandQueue!
    var pipeline: MTLRenderPipelineState
    
    private var positionsBuffer : MTLBuffer
    private var velocitiesBuffer : MTLBuffer
        
    private var transformBuffer : MTLBuffer
    
    override init() {
        
        let device = MTLCreateSystemDefaultDevice()!
        
        var graph = Graph(graphParams: .init(tileSize: 0.1, numTiles: 10))
        let numNodes = graph.numNodes.intValue
        graph.buildGraphStandard(sparse: false)
        
        //graph.buildGraphReducedPrecision()

        //graph.buildGraphNew()
        graph.compile(on: device, useFloat16: false)
        

        let positions = Self.buildRandomPositions(numNodes: numNodes, of: Float32.self)
        
        let positionsBuffer = device.makeBuffer(bytes: consume positions,
                                                length: MemoryLayout<SIMD2<Float32>>.size * numNodes,
                                                options: .storageModeShared)!
        
        let velocitiesBuffer = device.makeBuffer(length: MemoryLayout<SIMD2<Float32>>.size * numNodes,
                                                 options: .storageModeShared)!
        
        let transformBuffer = device.makeBuffer(length: MemoryLayout<ScreenTransform>.size, options: .cpuCacheModeWriteCombined)!
                
        self.graph = consume graph
        
        self.positionsBuffer = consume positionsBuffer
        self.velocitiesBuffer = consume velocitiesBuffer
        self.transformBuffer = consume transformBuffer
        
        self.metalDevice = device
        
        guard let library = try? device.makeDefaultLibrary(bundle: .module) else {
            fatalError("failed to load library")
        }
        
        let vertexDescriptor = Self.makeNodeVertexDescriptor()
        
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = library.makeFunction(name: "vertexMain")
        pipelineDescriptor.fragmentFunction = library.makeFunction(name: "fragmentMain")
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        
        pipelineDescriptor.vertexDescriptor = vertexDescriptor
        
        guard let pipeline = try? device.makeRenderPipelineState(descriptor: pipelineDescriptor) else {
            fatalError("failed to create pipeline state")
        }
        self.pipeline = pipeline
        
        self.commandQueue = device.makeCommandQueue()
        
        super.init()
    }
    
    static func buildStructuredPositions<N: FloatingPoint & SIMDScalar>(
        numNodes: Int,
        of type: N.Type,
        numSteps: Int = 4
    ) -> [SIMD2<N>] {
        let positions: [SIMD2<N>] = .init(unsafeUninitializedCapacity: numNodes) { buffer, initializedCount in
            let max: N = 1
            let gridSize = numSteps * numSteps  // number of regions
            let nodesPerRegion = (numNodes + gridSize - 1) / gridSize  // ceil division

            var idx = 0
            for region in 0..<gridSize {
                let gx = region % numSteps
                let gy = region / numSteps

                for j in 0..<nodesPerRegion {
                    guard idx < numNodes else { break }

                    // Spread nodes inside each region slightly
                    let localX: N = N(j % nodesPerRegion) * (max / N(nodesPerRegion))
                    let localY: N = N(j / nodesPerRegion) * (max / N(nodesPerRegion))

                    // Region center offset
                    let x = (N(gx) - N(numSteps - 1) / 2) * max + localX
                    let y = (N(gy) - N(numSteps - 1) / 2) * max + localY

                    buffer[idx] = SIMD2<N>(x, y)
                    idx += 1
                    //print(x,y)
                }
            }
            initializedCount = numNodes
        }
        print(positions)
        return positions
    }
    
    static func buildRandomPositions<N: FloatingPoint & SIMDScalar>(numNodes: Int, of type: N.Type) -> [SIMD2<N>] {
        let positions: [SIMD2<N>] = .init(unsafeUninitializedCapacity: numNodes) { buffer, initializedCount in
            for idx in 0..<numNodes {

                if N.self == Float32.self {
                    let x = Float32.random(in: -0.5...0.5) as! N
                    let y = Float32.random(in: -0.5...0.5) as! N
                    
                    buffer[idx] = .init(x: x, y: y)
                } else if N.self == Float16.self {
                    let x = Float16.random(in: -0.5...0.5) as! N
                    let y = Float16.random(in: -0.5...0.5) as! N
                    
                    buffer[idx] = .init(x: x, y: y)
                } else {
                    fatalError("ASHDFASJDHFLAJKSDFLKAJSDFLKAJSDF")
                }
            }
            initializedCount = numNodes
        }
        print(positions)
        return positions
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        print("drawableSize", view.drawableSize)
        print("will change", size)
        //print("preferredDrawableSize", view.preferredDrawableSize)
    }
    
    func draw(in view: MTKView) {
        
        guard let drawable = view.currentDrawable,
              let descriptor = view.currentRenderPassDescriptor else { return }
        
        let commandBuffer = self.commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor)!
        
        self.graph.runInplace(positionsBuffer: &self.positionsBuffer, velocitiesBuffer: &self.velocitiesBuffer, on: self.commandQueue)
        //self.graph.testRun(positionsBuffer: &self.positionsBuffer, velocitiesBuffer: &self.velocitiesBuffer)
        
        var transform : ScreenTransform = .init()
        
        memcpy(self.transformBuffer.contents(), &transform, MemoryLayout<ScreenTransform>.stride)

        
        encoder.setRenderPipelineState(self.pipeline)
        encoder.setVertexBuffer(self.positionsBuffer, offset: 0, index: 0)
        encoder.setVertexBuffer(self.transformBuffer, offset: 0, index: 1)
        encoder.drawPrimitives(type: .point,
                               vertexStart: 0,
                               vertexCount: self.graph.numNodes.intValue)
        
        encoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
    
    private static func makeNodeVertexDescriptor() -> MTLVertexDescriptor {
        let vertexDescriptor = MTLVertexDescriptor()

        // Attribute 0 = position
        vertexDescriptor.attributes[0].format = .float2        // two floats (x,y)
        vertexDescriptor.attributes[0].offset = 0              // starts at byte 0
        vertexDescriptor.attributes[0].bufferIndex = 0         // comes from buffer(0)

        // Layout for buffer(0)
        vertexDescriptor.layouts[0].stride = MemoryLayout<SIMD2<Float32>>.stride
        vertexDescriptor.layouts[0].stepFunction = .perVertex
        
        
        return vertexDescriptor
    }
    
}

