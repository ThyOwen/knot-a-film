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
    
    // Tree construction parameters
    let maxDepth: Int32 = 9
    static let nLeaf: Int32 = 262144
    
    var numBodies : Int32 = NUM_BODIES
    var numNodes : Int32 = MAX_NODES
    var leafLimit : Int32 = MAX_NODES - nLeaf
    var maxConnections : Int32 = MAX_CONNECTIONS
    
    let minDist : Float = 0.3
    let maxDist : Float = 0.8
    
    let centerX : Float = 0.0
    let centerY : Float = 0.0
    
    let sunMass : Float = 10.0
    let sunDiameter : Float = 0.05
    let earthMass : Float = 1.0
    let earthDiameter : Float = 0.01
    
    let blockSize : Int32 = 512

    var physics = PhysicsParams(
        gravity: 0.0001,
        epsilon: 0.01,
        dt: 0.016,
        theta: 0.5,
        collisionThreshold: 0.0,
        damping: 0.98
    )
    
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
        
        let params = GraphParams()
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
        
        computeEncoder.setBytes(&self.params.physics, length: MemoryLayout<PhysicsParams>.stride, index: 2)
        
        // Set threadgroup memory for tileB array
        let tileBlockMemSize = MemoryLayout<Body>.stride * Int(self.params.blockSize)
        computeEncoder.setThreadgroupMemoryLength(tileBlockMemSize, index: 0)
    
        let threadsPerThreadgroup = MTLSize(width: Int(self.params.blockSize), height: 1, depth: 1)
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
        let maxDistance = params.maxDist
        let minDistance = params.minDist
        let numBodies = Int(params.numBodies)
        
        let centerPos = SIMD2<Float32>(x: params.centerX, y: params.centerY)

        var bodies : [Body] = .init(repeating: Body(), count: numBodies)
        
        for i in 0..<(numBodies - 1) {
            let angle : Float32 = 2.0 * Float32.pi * Float32.random(in: 0...1)
            let radius : Float32 = (maxDistance - minDistance) * Float32.random(in: 0...1) + minDistance

            let x = centerPos.x + radius * Float32(cos(angle))
            let y = centerPos.y + radius * Float32(sin(angle))
            let position = SIMD2<Float32>(x: x, y: y)
            
            let body : Body = .init(
                isDynamic: true,
                mass: params.earthMass,
                radius: params.earthDiameter,
                position: position,
                velocity: SIMD2<Float32>(x: 0.0, y: 0.0),
                acceleration: SIMD2<Float32>(x: 0.0, y: 0.0)
            )
            bodies[i] = body
        }

        let body : Body = .init(
            isDynamic: false,
            mass: params.sunMass,
            radius: params.sunDiameter,
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
    
    private var coalesceConnectionsIndicesPipeline : MTLComputePipelineState
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
    private var coalescedIndicesBuffer : MTLBuffer
    
    private var params : GraphParams
    let numConnections : Int


    public var screenTransform : ScreenTransform = .init(offset: .zero, zoom: .init(1.0, 1.0))

    private let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    
    public init(connections : [SIMD2<UInt32>]) {

        var params = GraphParams()
        // Total nodes = 1 + 4 + 16 + ... + 4^maxDepth = (4^(maxDepth+1) - 1) / 3
        params.numNodes = (Int32(pow(4.0, Double(params.maxDepth + 1))) - 1) / 3
        params.leafLimit = (Int32(pow(4.0, Double(params.maxDepth))) - 1) / 3
        self.params = params
        self.numConnections = connections.count
        
        let device = MTLCreateSystemDefaultDevice()!
        
        let bodies = GraphRendererDirectSum.initRandomBodies(params: params)
        
        let bodySize = MemoryLayout<BodyData>.stride
        self.bodyBuffer = device.makeBuffer(length: bodySize, options: .storageModeShared)!
        self.bodyBufferAlt = device.makeBuffer(length: bodySize, options: .storageModeShared)!
        
        //Bodies
        let bodyDataPtr = bodyBuffer.contents().bindMemory(to: BodyData.self, capacity: 1)
        withUnsafeMutablePointer(to: &bodyDataPtr.pointee.bodies.0) { bodiesPtr in
            for (i, body) in bodies.enumerated() where i < Int(params.numBodies) {
                bodiesPtr.advanced(by: i).pointee = body
            }
        }
        bodyDataPtr.pointee.numBodies = uint(params.numBodies)
        
        let bodyDataAltPtr = bodyBufferAlt.contents().bindMemory(to: BodyData.self, capacity: 1)
        withUnsafeMutablePointer(to: &bodyDataAltPtr.pointee.bodies.0) { bodiesPtr in
            for (i, body) in bodies.enumerated() where i < Int(params.numBodies) {
                bodiesPtr.advanced(by: i).pointee = body
            }
        }
        bodyDataAltPtr.pointee.numBodies = uint(params.numBodies)
        
        //Nodes
        let nodeSize = MemoryLayout<NodeData>.stride
        self.nodeBuffer = device.makeBuffer(length: nodeSize, options: .storageModeShared)!
        let nodeDataPtr = nodeBuffer.contents().bindMemory(to: NodeData.self, capacity: 1)
        nodeDataPtr.pointee.numNodes = uint(params.numNodes)
        
        //Connections
        let coalescedIndicesSize = MemoryLayout<ConnectionsData>.stride
        self.coalescedIndicesBuffer = device.makeBuffer(length: coalescedIndicesSize, options: .storageModeShared)!
        
        if connections.isEmpty {
            self.connectionsBuffer = device.makeBuffer(length: MemoryLayout<SIMD2<UInt32>>.stride, options: .storageModeShared)!
        } else {
            let connectionsBufferSize = connections.count * MemoryLayout<SIMD2<UInt32>>.stride
            self.connectionsBuffer = device.makeBuffer(bytes: consume connections, length: connectionsBufferSize, options: .storageModeShared)!
        }

        //Helpers
        let mutexSize = MemoryLayout<Int32>.stride * Int(params.numNodes)
        self.mutexBuffer = device.makeBuffer(length: mutexSize, options: .storageModeShared)!
        
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
        
        let coalesceConnectionsIndicesFunction = library.makeFunction(name: "coalesceConnectionsIndices")!
        self.coalesceConnectionsIndicesPipeline = try! device.makeComputePipelineState(function: coalesceConnectionsIndicesFunction)
        
        let resetFunction = library.makeFunction(name: "resetKernel")!
        self.resetPipeline = try! device.makeComputePipelineState(function: resetFunction)
        
        let boundingBoxFunction = library.makeFunction(name: "computeBoundingBoxKernel")!
        self.boundingBoxPipeline = try! device.makeComputePipelineState(function: boundingBoxFunction)
        
        let computeForceKernel = library.makeFunction(name: "computeForceKernel")!
        self.computeForcePipeline = try! device.makeComputePipelineState(function: computeForceKernel)
        
        let constructTreeFunction = library.makeFunction(name: "constructQuadTreeKernel")!
        self.constructTreePipeline = try! device.makeComputePipelineState(function: constructTreeFunction)
        
        self.commandQueue = device.makeCommandQueue()!

        self.device = device
        super.init()
    }
    
    private func coalesceConnectionsIndices(commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        encoder.setComputePipelineState(coalesceConnectionsIndicesPipeline)
        
        encoder.setBuffer(connectionsBuffer, offset: 0, index: 0)
        encoder.setBuffer(coalescedIndicesBuffer, offset: 0, index: 1)
        var numConns = Int32(self.numConnections)
        encoder.setBytes(&numConns, length: MemoryLayout<Int32>.stride, index: 2)
        encoder.setBytes(&params.numBodies, length: MemoryLayout<Int32>.stride, index: 3)
        
        // Set threadgroup memory for connectionsTile array
        let connectionsTileMemSize = MemoryLayout<SIMD2<UInt32>>.stride * Int(params.blockSize)
        encoder.setThreadgroupMemoryLength(connectionsTileMemSize, index: 0)

        let gridSize = MTLSize(width: (Int(params.numBodies) + threadgroupSize.width - 1) / threadgroupSize.width,
                              height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        
        encoder.endEncoding()

    }

    private func resetTree(commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        encoder.setComputePipelineState(resetPipeline)
        encoder.setBuffer(nodeBuffer, offset: 0, index: 0)
        encoder.setBuffer(mutexBuffer, offset: 0, index: 1)
        encoder.setBuffer(bodyBuffer, offset: 0, index: 2)
        
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
        
        encoder.setBuffer(nodeBuffer, offset: 0, index: 0)
        encoder.setBytes(&params.leafLimit, length: MemoryLayout<Int32>.stride, index: 4)
        
        for level in 0...self.params.maxDepth {
            let nodesInLevel = Int(pow(4.0, Double(level)))
            var nodeOffset = (Int32(pow(4.0, Double(level))) - 1) / 3
                        
            encoder.setBuffer(currentBuffer, offset: 0, index: 1)
            encoder.setBuffer(nextBuffer, offset: 0, index: 2)
            encoder.setBytes(&nodeOffset, length: MemoryLayout<Int32>.stride, index: 3)


            let countMemSize = MemoryLayout<Int32>.stride * 8
            let massMemSize = MemoryLayout<Float>.stride * threadgroupSize.width
            let centerMemSize = MemoryLayout<SIMD2<Float32>>.stride * threadgroupSize.width
            encoder.setThreadgroupMemoryLength(countMemSize, index: 0)
            encoder.setThreadgroupMemoryLength(massMemSize, index: 1)
            encoder.setThreadgroupMemoryLength(centerMemSize, index: 2)
            
            let gridSize = MTLSize(width: nodesInLevel, height: 1, depth: 1)
            encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
            
            swap(&currentBuffer, &nextBuffer)
        }
        
        encoder.endEncoding()
        
        if self.params.maxDepth % 2 == 1 {
            swap(&bodyBuffer, &bodyBufferAlt)
        }
    }
    
    private func computeForces(commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        encoder.setComputePipelineState(computeForcePipeline)
        encoder.setBuffer(nodeBuffer, offset: 0, index: 0)
        encoder.setBuffer(coalescedIndicesBuffer, offset: 0, index: 1)
        encoder.setBuffer(bodyBuffer, offset: 0, index: 2)
        encoder.setBytes(&params.leafLimit, length: MemoryLayout<Int32>.stride, index: 3)
        
        encoder.setBytes(&self.params.physics, length: MemoryLayout<PhysicsParams>.stride, index: 4)
        
        // Set threadgroup memory for tileBlock array
        let tileBlockMemSize = MemoryLayout<Body>.stride * Int(params.blockSize)
        encoder.setThreadgroupMemoryLength(tileBlockMemSize, index: 0)
        
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
        self.coalesceConnectionsIndices(commandBuffer: commandBuffer)
        self.resetTree(commandBuffer: commandBuffer)
        self.computeBoundingBox(commandBuffer: commandBuffer)
        self.constructQuadTree(commandBuffer: commandBuffer)
        self.computeForces(commandBuffer: commandBuffer)
        
        //computeEncoder.endEncoding()
        
        memcpy(self.transformBuffer.contents(), &self.screenTransform, MemoryLayout<ScreenTransform>.stride)

        let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor)!
    
        renderEncoder.setVertexBuffer(self.transformBuffer, offset: 0, index: 1)
        
        // draw bounding box nodes
        renderEncoder.setRenderPipelineState(self.nodeRenderPipeline)
        renderEncoder.setVertexBuffer(self.nodeBuffer, offset: 0, index: 0)
        renderEncoder.drawPrimitives(type: .line, vertexStart: 0, vertexCount: Int(self.params.numNodes) * 8)
        
        // draw bodies 
        renderEncoder.setRenderPipelineState(self.bodyRenderPipeline)
        renderEncoder.setVertexBuffer(self.bodyBuffer, offset: 0, index: 0)
        renderEncoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: Int(self.params.numBodies))
    
        renderEncoder.endEncoding()
        commandBuffer.present(drawable)
        
        commandBuffer.commit()
        
        //commandBuffer.waitUntilCompleted()
        
        //self.printTreeDebug()
        
        //self.printConnectionIdxDebug()
    }
    
    
    private func printTreeDebug() {
        let nodePtr = nodeBuffer.contents().assumingMemoryBound(to: Node.self)
        
        var leafCount = 0, internalCount = 0, emptyCount = 0
        
        print("\n=== TREE STRUCTURE DEBUG ===")
        for i in 0..<min(21, Int(self.params.numNodes)) {
            let node = nodePtr[i]
            let isEmpty = node.start == UInt32.max
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
            let node = nodePtr[i]
            if node.start == UInt32.max { emptyCount += 1 }
            else if node.isLeaf { leafCount += 1 }
            else { internalCount += 1 }
        }
        
        print("TOTAL: \(leafCount) leaves, \(internalCount) internal, \(emptyCount) empty")
    }
    
    private func printConnectionIdxDebug() {
        let uint32Ptr = coalescedIndicesBuffer.contents().assumingMemoryBound(to: UInt32.self)
        let perBodyDataSize = 512 + 1  // 512 connection slots + 1 count field
        
        print("\n=== CONNECTIONS DEBUG ===")
        for bodyIdx in 0..<min(5, Int(self.params.numBodies)) {
            let bodyOffset = bodyIdx * perBodyDataSize
            let numConns = Int(uint32Ptr[bodyOffset + 512])
            
            if numConns == 0 { continue }
            
            print("Body[\(bodyIdx)] has \(numConns) connections:")
            for i in 0..<min(numConns, 30) {
                let connIdx = uint32Ptr[bodyOffset + i]
                print("  -> Body[\(connIdx)]")
            }
        }

    }
}
