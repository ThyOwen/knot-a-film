//
//  GraphNew.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 9/7/25.
//

import MetalKit
import SharedWithMetal

public struct GraphParams {
    
    let edgeRepulsion : Double = 0.10
    let edgeAttraction : Double = 0.1
    let damping : Double = 0.001
    let epsilon : Double = 1e-03
    
    var numBodies : Int32
    var numNodes : Int32 = MAX_NODES
    var leafLimit : Int32 = MAX_NODES - N_LEAF
    let maxDepth: Int32 = MAX_DEPTH
        
    init(_ numBodies : Int32) {
        self.numBodies = numBodies
    }
}


public final class GraphRendererDirectSum : NSObject, MTKViewDelegate {
    
    let metalDevice : MTLDevice
    let commandQueue: MTLCommandQueue!
    let renderPipeline: MTLRenderPipelineState
    
    let forcePSO : MTLComputePipelineState
    
    public var bodiesBuffer : MTLBuffer

    public var transformBuffer : MTLBuffer
    public var connectionsBuffer : MTLBuffer
    
    
    public var params : GraphParams
    
    public var screenTransform : ScreenTransform = .init(offset: .zero, zoom: .init(1.0, 1.0))
        
    init(connections : [SIMD2<UInt32>]) {

        let device = MTLCreateSystemDefaultDevice()!
        
        let params = GraphParams(12000)
        self.params = params
        
        let bodies = Self.initRandomBodies(params: params)
        
        let bodiesBuffer = device.makeBuffer(bytes: consume bodies,
                                             length: MemoryLayout<Body>.size * Int(params.numBodies),
                                             options: .storageModeShared)!

        let connections = device.makeBuffer(bytes: connections,
                                            length: MemoryLayout<SIMD2<UInt32>>.size * connections.count,
                                            options: .storageModeShared)!
        
        let transformBuffer = device.makeBuffer(length: MemoryLayout<ScreenTransform>.size,
                                                options: .cpuCacheModeWriteCombined)!
                        
        self.bodiesBuffer = consume bodiesBuffer

        
        self.connectionsBuffer = consume connections
        self.transformBuffer = consume transformBuffer
        
        self.metalDevice = device
        
        let vertexDescriptor = Self.makeBodyVertexDescriptor()
        
        let library = try! device.makeDefaultLibrary(bundle: .main)
        
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = library.makeFunction(name: "vertexBody")
        pipelineDescriptor.fragmentFunction = library.makeFunction(name: "fragmentBody")
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        pipelineDescriptor.vertexDescriptor = vertexDescriptor
        
        self.renderPipeline = try! device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        
        let computeForceKernel = library.makeFunction(name: "directSumTiledKernel")!
        self.forcePSO = try! device.makeComputePipelineState(function: computeForceKernel)
        
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
        
        let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
        
        computeEncoder.setComputePipelineState(self.forcePSO)
        computeEncoder.setBuffer(self.bodiesBuffer, offset: 0, index: 0)
        computeEncoder.setBytes(&self.params.numBodies, length: MemoryLayout<Int32>.size, index: 1)
    
        let threadsPerThreadgroup = MTLSize(width: 128, height: 1, depth: 1)
        let threadsPerGrid = MTLSize(width: Int(self.params.numBodies), height: 1, depth: 1)
        
        computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        
        computeEncoder.endEncoding()
        let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor)!
        
        memcpy(self.transformBuffer.contents(), &self.screenTransform, MemoryLayout<ScreenTransform>.stride)

        renderEncoder.setRenderPipelineState(self.renderPipeline)
        renderEncoder.setVertexBuffer(self.bodiesBuffer, offset: 0, index: 0)
        renderEncoder.setVertexBuffer(self.transformBuffer, offset: 0, index: 1)
        renderEncoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: Int(self.params.numBodies))
        
        renderEncoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
        
        commandBuffer.waitUntilCompleted()
        
        let ptrBody = bodiesBuffer.contents().bindMemory(to: Body.self, capacity: Int(self.params.numBodies))
        print("=== Frame Update (\(self.params.numBodies) bodies) ===")
        for i in 0..<Int(self.params.numBodies) {
            print("Body \(i):", ptrBody[i])
        }
    }

    public static func makeBodyVertexDescriptor() -> MTLVertexDescriptor {
        let vertexDescriptor = MTLVertexDescriptor()

        vertexDescriptor.attributes[0].format = .float2
        vertexDescriptor.attributes[0].offset = 16
        vertexDescriptor.attributes[0].bufferIndex = 0

        vertexDescriptor.layouts[0].stride = MemoryLayout<Body>.stride
        vertexDescriptor.layouts[0].stepFunction = .perVertex

        return vertexDescriptor
    }

    static public func initRandomBodies(params : borrowing GraphParams) -> [Body] {
        let maxDistance: Float = Float(MAX_DIST)
        let minDistance: Float = Float(MIN_DIST)
        let numBodies = Int(params.numBodies)
        
        let centerPos = SIMD2<Float32>(x: Float32(CENTERX), y: Float32(CENTERY))

        var bodies : [Body] = .init(repeating: Body(), count: numBodies)
        
        for i in 0..<(numBodies - 1) {
            let angle : Float32 = 2.0 * Float32.pi * Float32.random(in: 0...1)
            let radius : Float32 = (maxDistance - minDistance) * Float32.random(in: 0...1) + minDistance

            let x = centerPos.x + radius * cos(angle)
            let y = centerPos.y + radius * sin(angle)
            let position = SIMD2<Float32>(x: x, y: y)
            
            let body : Body = .init(
                isDynamic: true,
                mass: 1,//Float(EARTH_MASS),
                radius: 1,//Float(EARTH_DIA),
                position: position,
                velocity: SIMD2<Float32>(x: 0.0, y: 0.0),
                acceleration: SIMD2<Float32>(x: 0.0, y: 0.0)
            )
            bodies[i] = body
        }

        let body : Body = .init(
            isDynamic: false,
            mass: Float(SUN_MASS),
            radius: Float(SUN_DIA),
            position: centerPos,
            velocity: SIMD2<Float32>(x: 0.0, y: 0.0),
            acceleration: SIMD2<Float32>(x: 0.0, y: 0.0)
        )
        
        bodies[numBodies - 1] = body
        
        return bodies
    }

}


public final class GraphRenderer : NSObject, MTKViewDelegate {
    
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    
    private var resetPipeline : MTLComputePipelineState
    private var boundingBoxPipeline : MTLComputePipelineState
    private var constructTreePipeline : MTLComputePipelineState
    private var computeForcePipeline : MTLComputePipelineState
    
    private var nodeRenderPipeline: MTLRenderPipelineState
    private var bodyRenderPipeline: MTLRenderPipelineState
    
    private var nodeBuffer : MTLBuffer
    private var bodyBuffer : MTLBuffer
    private var bodyBufferAlt : MTLBuffer  // Ping-pong buffer
    private var mutexBuffer : MTLBuffer
    private var transformBuffer : MTLBuffer
    
    public var connectionsBuffer : MTLBuffer
    
    private var params : GraphParams

    public var screenTransform : ScreenTransform = .init(offset: .zero, zoom: .init(1.0, 1.0))

    private let threadgroupSize = MTLSize(width: 576, height: 1, depth: 1)
    
    public init(connections : [SIMD2<UInt32>]) {

        var params = GraphParams(10000)
        // Total nodes = 1 + 4 + 16 + ... + 4^maxDepth = (4^(maxDepth+1) - 1) / 3
        params.numNodes = (Int32(pow(4.0, Double(params.maxDepth + 1))) - 1) / 3
        params.leafLimit = (Int32(pow(4.0, Double(params.maxDepth))) - 1) / 3
        self.params = params
        
        let device = MTLCreateSystemDefaultDevice()!
        
        let bodies = GraphRendererDirectSum.initRandomBodies(params: params)
        
        let bodySize = MemoryLayout<Body>.stride * Int(params.numBodies)
        self.bodyBuffer = device.makeBuffer(bytes: bodies, length: bodySize, options: .storageModeManaged)!
        self.bodyBufferAlt = device.makeBuffer(length: bodySize, options: .storageModePrivate)!
        
        let nodeSize = MemoryLayout<Node>.stride * Int(params.numNodes)
        self.nodeBuffer = device.makeBuffer(length: nodeSize, options: .storageModePrivate)!

        let mutexSize = MemoryLayout<Int32>.stride * Int(params.numNodes)
        self.mutexBuffer = device.makeBuffer(length: mutexSize, options: .storageModePrivate)!
        
        let transformSize = MemoryLayout<ScreenTransform>.size
        self.transformBuffer = device.makeBuffer(length: transformSize,
                                                options: .cpuCacheModeWriteCombined)!

        let library = try! device.makeDefaultLibrary(bundle: .main)
        
        //render
        let vertexDescriptor = GraphRendererDirectSum.makeBodyVertexDescriptor()
        
        let bodyPipelineDescriptor = MTLRenderPipelineDescriptor()
        bodyPipelineDescriptor.vertexFunction = library.makeFunction(name: "vertexBody")
        bodyPipelineDescriptor.fragmentFunction = library.makeFunction(name: "fragmentBody")
        bodyPipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        
        let nodePipelineDescriptor = MTLRenderPipelineDescriptor()
        nodePipelineDescriptor.vertexFunction = library.makeFunction(name: "vertexNode")
        nodePipelineDescriptor.fragmentFunction = library.makeFunction(name: "fragmentNode")
        nodePipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        
        self.nodeRenderPipeline = try! device.makeRenderPipelineState(descriptor: nodePipelineDescriptor)
        self.bodyRenderPipeline = try! device.makeRenderPipelineState(descriptor: bodyPipelineDescriptor)

        //compute
        let resetFunction = library.makeFunction(name: "resetKernel")!
        self.resetPipeline = try! device.makeComputePipelineState(function: resetFunction)
        
        let boundingBoxFunction = library.makeFunction(name: "computeBoundingBoxKernel")!
        self.boundingBoxPipeline = try! device.makeComputePipelineState(function: boundingBoxFunction)
        
        let computeForceKernel = library.makeFunction(name: "computeForceKernel")!
        self.computeForcePipeline = try! device.makeComputePipelineState(function: computeForceKernel)
        
        let constructTreeFunction = library.makeFunction(name: "constructQuadTreeKernel")!
        self.constructTreePipeline = try! device.makeComputePipelineState(function: constructTreeFunction)
        
        self.commandQueue = device.makeCommandQueue()!

        self.connectionsBuffer = device.makeBuffer(length: bodySize, options: .storageModePrivate)!
        self.device = device
        super.init()
    }

    private func resetTree(commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        encoder.setComputePipelineState(resetPipeline)
        encoder.setBuffer(nodeBuffer, offset: 0, index: 0)
        encoder.setBuffer(mutexBuffer, offset: 0, index: 1)
        
        encoder.setBytes(&params.numNodes, length: MemoryLayout<Int32>.stride, index: 2)
        encoder.setBytes(&params.numBodies, length: MemoryLayout<Int32>.stride, index: 3)
        
        let gridSize = MTLSize(width: (Int(params.numNodes) + threadgroupSize.width - 1) / threadgroupSize.width,
                              height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
    
    private func computeBoundingBox(commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        encoder.setComputePipelineState(boundingBoxPipeline)
        encoder.setBuffer(nodeBuffer, offset: 0, index: 0)
        encoder.setBuffer(bodyBuffer, offset: 0, index: 1)
        encoder.setBuffer(mutexBuffer, offset: 0, index: 2)
        
        encoder.setBytes(&self.params.numBodies, length: MemoryLayout<Int32>.stride, index: 3)
        
        let threadgroupMemSize = MemoryLayout<Float>.stride * threadgroupSize.width
        encoder.setThreadgroupMemoryLength(threadgroupMemSize, index: 0) // topLeftX
        encoder.setThreadgroupMemoryLength(threadgroupMemSize, index: 1) // topLeftY
        encoder.setThreadgroupMemoryLength(threadgroupMemSize, index: 2) // botRightX
        encoder.setThreadgroupMemoryLength(threadgroupMemSize, index: 3) // botRightY
        
        let gridSize = MTLSize(width: (Int(params.numBodies) + threadgroupSize.width - 1) / threadgroupSize.width,
                              height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
    
    private func constructQuadTree(commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        encoder.setComputePipelineState(constructTreePipeline)
        
        var currentBuffer = bodyBuffer
        var nextBuffer = bodyBufferAlt
        
        for level in 0...self.params.maxDepth {
            let nodesInLevel = Int(pow(4.0, Double(level)))
            var nodeOffset = (Int32(pow(4.0, Double(level))) - 1) / 3
                        
            encoder.setBuffer(nodeBuffer, offset: 0, index: 0)
            encoder.setBuffer(currentBuffer, offset: 0, index: 1)
            encoder.setBuffer(nextBuffer, offset: 0, index: 2)
            encoder.setBytes(&nodeOffset, length: MemoryLayout<Int32>.stride, index: 3)
            encoder.setBytes(&params.numNodes, length: MemoryLayout<Int32>.stride, index: 4)
            encoder.setBytes(&params.numBodies, length: MemoryLayout<Int32>.stride, index: 5)
            encoder.setBytes(&params.leafLimit, length: MemoryLayout<Int32>.stride, index: 6)
            
            let countMemSize = MemoryLayout<Int32>.stride * 8
            let massMemSize = MemoryLayout<Float>.stride * threadgroupSize.width
            let centerMemSize = MemoryLayout<SIMD2<Float>>.stride * threadgroupSize.width
            encoder.setThreadgroupMemoryLength(countMemSize, index: 0)
            encoder.setThreadgroupMemoryLength(massMemSize, index: 1)
            encoder.setThreadgroupMemoryLength(centerMemSize, index: 2)
            
            let gridSize = MTLSize(width: nodesInLevel, height: 1, depth: 1)
            encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
            
            swap(&currentBuffer, &nextBuffer)
        }
        
        encoder.endEncoding()
        
        if self.params.maxDepth % 2 == 1 {// If we did an odd number of swaps, the final data is in bodyBufferAlt
            swap(&bodyBuffer, &bodyBufferAlt)
        }
    }
    
    private func computeForces(commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        encoder.setComputePipelineState(computeForcePipeline)
        encoder.setBuffer(nodeBuffer, offset: 0, index: 0)
        encoder.setBuffer(bodyBuffer, offset: 0, index: 1)
        
        encoder.setBytes(&params.numNodes, length: MemoryLayout<Int32>.stride, index: 2)
        encoder.setBytes(&params.numBodies, length: MemoryLayout<Int32>.stride, index: 3)
        encoder.setBytes(&params.leafLimit, length: MemoryLayout<Int32>.stride, index: 4)
        
        let gridSize = MTLSize(width: (Int(params.numBodies) + threadgroupSize.width - 1) / threadgroupSize.width,
                              height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
    
    // MARK: - MTKViewDelegate
    
    public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {

    }
    
    public func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable, let descriptor = view.currentRenderPassDescriptor else {
            return
        }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        
        //let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
        
        self.resetTree(commandBuffer: commandBuffer)
        self.computeBoundingBox(commandBuffer: commandBuffer)
        self.constructQuadTree(commandBuffer: commandBuffer)
        self.computeForces(commandBuffer: commandBuffer)
        
        //computeEncoder.endEncoding()
        
        memcpy(self.transformBuffer.contents(), &self.screenTransform, MemoryLayout<ScreenTransform>.stride)

        let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor)!
    
        // draw bounding box nodes
        renderEncoder.setRenderPipelineState(self.nodeRenderPipeline)
        renderEncoder.setVertexBuffer(self.nodeBuffer, offset: 0, index: 0)
        renderEncoder.setVertexBuffer(self.transformBuffer, offset: 0, index: 1)
        renderEncoder.drawPrimitives(type: .line, vertexStart: 0, vertexCount: Int(self.params.numNodes) * 8)
        
        // draw bodies 
        renderEncoder.setRenderPipelineState(self.bodyRenderPipeline)
        renderEncoder.setVertexBuffer(self.bodyBuffer, offset: 0, index: 0)
        renderEncoder.setVertexBuffer(self.transformBuffer, offset: 0, index: 1)
        renderEncoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: Int(self.params.numBodies))
    
        renderEncoder.endEncoding()
        commandBuffer.present(drawable)
        
        commandBuffer.commit()
        
        /*
        commandBuffer.waitUntilCompleted()

         
        let ptrNode = nodeBuffer.contents().bindMemory(to: Node.self, capacity: Int(self.params.numNodes))

        var leafCount = 0, internalCount = 0, emptyCount = 0
        
        print("\n=== TREE STRUCTURE DEBUG ===")
        for i in 0..<Int(self.params.numNodes) {
            let node = ptrNode[i]
            let isEmpty = node.start == UInt32.max  // -1 stored as unsigned
            let statusStr = isEmpty ? "EMPTY" : (node.isLeaf ? "LEAF" : "INTERNAL")
            print("Node[\(i)]: \(statusStr), start=\(node.start), end=\(node.end), mass=\(node.totalMass), COM=(\(node.centerOfMass.x), \(node.centerOfMass.y))")
            print("         bounds: topLeft=(\(node.topLeft.x), \(node.topLeft.y)), bottomRight=(\(node.bottomRight.x), \(node.bottomRight.y))")
            
            if isEmpty {
                emptyCount += 1
            } else if node.isLeaf {
                leafCount += 1
            } else {
                internalCount += 1
            }
        }
        
        for i in 21..<Int(self.params.numNodes) {
            let node = ptrNode[i]
            if node.start == UInt32.max { emptyCount += 1 }
            else if node.isLeaf { leafCount += 1 }
            else { internalCount += 1 }
        }
        
        print("TOTAL: \(leafCount) leaves, \(internalCount) internal, \(emptyCount) empty")
        */
    }
}
