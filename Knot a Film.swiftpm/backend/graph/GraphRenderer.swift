//
//  GraphNew.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 9/7/25.
//

import MetalKit

public struct GraphParams {
    
    struct TileParams<N : Numeric & FloatingPoint> {
        let tileSize : N
        let numTiles : Int
    }
    
    let widthParams : TileParams<Float32>
    let heightParams : TileParams<Float32>
    
    let edgeRepulsion : Double = 0.10
    let edgeAttraction : Double = 0.1
    let damping : Double = 0.001
    let epsilon : Double = 1e-03
    
    let numNodes : Int = 1000
    let useFloat16 : Bool = false
        
    init(tileSize: Float32, numTiles: Int) {
        self.widthParams = .init(tileSize: tileSize, numTiles: numTiles)
        self.heightParams = .init(tileSize: tileSize, numTiles: numTiles)
    }
}

public final class GraphRenderer : NSObject, MTKViewDelegate {

    var graph : Graph
    
    let metalDevice : MTLDevice
    let commandQueue: MTLCommandQueue!
    let pipeline: MTLRenderPipelineState
    
    public var positionsBuffer : MTLBuffer
    public var velocitiesBuffer : MTLBuffer
        
    public var transformBuffer : MTLBuffer
    public var connectionsBuffer : MTLBuffer
    
    public var screenTransform : ScreenTransform = .init()
        
    init(connections : [SIMD2<UInt32>]) {
        
        let device = MTLCreateSystemDefaultDevice()!
        
        var graph = Graph(graphParams: .init(tileSize: 0.1, numTiles: 10), connections: connections)
        let numNodes = graph.numNodes.intValue
        graph.buildGraphStandard()
        //graph.compile(on: device)
        
        let positions = Self.buildRandomPositions(numNodes: numNodes, of: Float32.self)
        
        let positionsBuffer = device.makeBuffer(bytes: consume positions,
                                                length: MemoryLayout<SIMD2<Float32>>.size * numNodes,
                                                options: .storageModeShared)!
        
        let velocitiesBuffer = device.makeBuffer(length: MemoryLayout<SIMD2<Float32>>.size * numNodes,
                                                 options: .storageModePrivate)!
        
        let connections = device.makeBuffer(bytes: connections,
                                            length: MemoryLayout<SIMD2<UInt32>>.size * connections.count,
                                            options: .storageModeShared)!
        
        let transformBuffer = device.makeBuffer(length: MemoryLayout<ScreenTransform>.size,
                                                options: .cpuCacheModeWriteCombined)!
                
        self.graph = consume graph
                
        self.positionsBuffer = consume positionsBuffer
        self.velocitiesBuffer = consume velocitiesBuffer
        
        self.connectionsBuffer = consume connections
        self.transformBuffer = consume transformBuffer
        
        self.metalDevice = device
        
        let vertexDescriptor = Self.makeNodeVertexDescriptor()
        
        guard let library = try? device.makeDefaultLibrary(bundle: .main) else {
            fatalError("failed to load library")
        }
        
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
    
    public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        print("drawableSize", view.drawableSize)
        print("will change", size)
        //print("preferredDrawableSize", view.preferredDrawableSize)
    }
    
    public func draw(in view: MTKView) {
        
        guard let drawable = view.currentDrawable,
              let descriptor = view.currentRenderPassDescriptor else {
            return
        }
        
        let commandBuffer = self.commandQueue.makeCommandBuffer()!
        
        let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor)!
        
        self.graph.runInplace(positionsBuffer: &self.positionsBuffer,
                              velocitiesBuffer: &self.velocitiesBuffer,
                              connectionsBuffer: &self.connectionsBuffer,
                              on: self.commandQueue)
        //self.graph.testRun(positionsBuffer: &self.positionsBuffer, velocitiesBuffer: &self.velocitiesBuffer)
        
        
        memcpy(self.transformBuffer.contents(), &self.screenTransform, MemoryLayout<ScreenTransform>.stride)

        
        encoder.setRenderPipelineState(self.pipeline)
        encoder.setVertexBuffer(self.positionsBuffer, offset: 0, index: 0)
        encoder.setVertexBuffer(self.transformBuffer, offset: 0, index: 1)
        encoder.drawPrimitives(type: .point,
                               vertexStart: 0,
                               vertexCount: self.graph.numNodes.intValue)
        
        encoder.drawIndexedPrimitives(
            type: .line,
            indexCount: self.graph.numConnections.intValue * 2,
            indexType: .uint32,
            indexBuffer: connectionsBuffer,
            indexBufferOffset: 0
        )

        
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

