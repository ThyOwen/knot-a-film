//
//  GraphNew.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 9/7/25.
//

import MetalKit
import SharedWithMetal

public struct GraphParams {
    
    let maxDepth: Int32 = 9
    static let nLeaf: Int32 = 262144
    
    var numBodies : Int32
    var numNodes : Int32 = MAX_NODES
    var leafLimit : Int32 = MAX_NODES - nLeaf
    var maxConnections : Int32 = MAX_CONNECTIONS
    
    let minDist : Float = 0.3
    let maxDist : Float = 0.8
    
    let centerX : Float = 0.0
    let centerY : Float = 0.0
    
    let blockSize : Int32 = 512

    var physics = PhysicsParams(
        springConstant: 2.0,
        edgeRepulsion: 100.0,
        edgeAttraction: 80.0,
        epsilon: 0.01,
        dt: 0.1,
        theta: 0.5,
        collisionThreshold: 0.001,
        damping: 0.95
    )
    
}

public final class GraphRenderer : NSObject, MTKViewDelegate {
    
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    
    private var initalizeBodiesPipeline : MTLComputePipelineState
    private var coalesceConnectionsIndicesPipeline : MTLComputePipelineState
    private var resetPipeline : MTLComputePipelineState
    private var boundingBoxPipeline : MTLComputePipelineState
    private var constructTreePipeline : MTLComputePipelineState
    private var computeForcePipeline : MTLComputePipelineState
    
    private var nodeRenderPipeline: MTLRenderPipelineState
    private var bodyRenderPipeline: MTLRenderPipelineState
    private var bodyLineRenderPipeline: MTLRenderPipelineState
    
    private var nodeBuffer : MTLBuffer
    private var bodyBuffer : MTLBuffer
    private var bodyBufferAlt : MTLBuffer  // Ping-pong buffer
    private var mutexBuffer : MTLBuffer
    private var transformBuffer : MTLBuffer
    
    public var connectionsBuffer : MTLBuffer
    
    private var params : GraphParams
    let numConnections : Int
    var bodiesInitialized: Bool = false

    public var screenTransform : ScreenTransform = .init(offset: .zero, scale: .init(1.0, 1.0))

    private let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
    
    public init(perBodyConnections : consuming [PerBodyConnectionsData], numConnections : Int32) {

        var params = GraphParams(numBodies: Int32(perBodyConnections.count))
        // Total nodes = 1 + 4 + 16 + ... + 4^maxDepth = (4^(maxDepth+1) - 1) / 3
        params.numNodes = (Int32(pow(4.0, Double(params.maxDepth + 1))) - 1) / 3
        params.leafLimit = (Int32(pow(4.0, Double(params.maxDepth))) - 1) / 3
        self.params = params
        self.numConnections = Int(numConnections)
        
        let device = MTLCreateSystemDefaultDevice()!
                
        let bodySize = MemoryLayout<BodyData>.stride
        self.bodyBuffer = device.makeBuffer(length: bodySize, options: .storageModeShared)!
        self.bodyBufferAlt = device.makeBuffer(length: bodySize, options: .storageModeShared)!
        
        //Bodies
        let bodyDataPtr = bodyBuffer.contents().bindMemory(to: BodyData.self, capacity: 1)
        bodyDataPtr.pointee.numBodies = uint(params.numBodies)
        
        let bodyDataAltPtr = bodyBufferAlt.contents().bindMemory(to: BodyData.self, capacity: 1)
        bodyDataAltPtr.pointee.numBodies = uint(params.numBodies)
        
        //Nodes
        let nodeSize = MemoryLayout<NodeData>.stride
        self.nodeBuffer = device.makeBuffer(length: nodeSize, options: .storageModeShared)!
        let nodeDataPtr = nodeBuffer.contents().bindMemory(to: NodeData.self, capacity: 1)
        nodeDataPtr.pointee.numNodes = uint(params.numNodes)
        
        //Connections
        let coalescedIndicesSize = MemoryLayout<ConnectionsData>.stride
        self.connectionsBuffer = device.makeBuffer(length: coalescedIndicesSize, options: .storageModeShared)!
        let connectionsDataPointer = connectionsBuffer.contents().bindMemory(to: ConnectionsData.self, capacity: 1)
        withUnsafeMutablePointer(to: &connectionsDataPointer.pointee.connections.0) { connectionsPointer in
            for (i, perBodyConnection) in perBodyConnections.enumerated() {
                connectionsPointer.advanced(by: i).pointee = perBodyConnection
            }
        }
        connectionsDataPointer.pointee.numConnections = uint(numConnections)
        print(numConnections)

        //Helpers
        let mutexSize = MemoryLayout<Int32>.stride * Int(params.numNodes)
        self.mutexBuffer = device.makeBuffer(length: mutexSize, options: .storageModePrivate)!
        
        let transformSize = MemoryLayout<ScreenTransform>.size
        self.transformBuffer = device.makeBuffer(length: transformSize,
                                                options: .cpuCacheModeWriteCombined)!

        let library = try! device.makeDefaultLibrary(bundle: .main)
        
        //render
        
        let nodePipelineDescriptor = MTLRenderPipelineDescriptor()
        nodePipelineDescriptor.vertexFunction = library.makeFunction(name: "vertexNode")
        nodePipelineDescriptor.fragmentFunction = library.makeFunction(name: "fragmentNode")
        nodePipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        
        
        let bodyPipelineDescriptor = MTLMeshRenderPipelineDescriptor()
        bodyPipelineDescriptor.meshFunction = library.makeFunction(name: "meshPointShader")
        bodyPipelineDescriptor.objectFunction = library.makeFunction(name: "objectShader")
        bodyPipelineDescriptor.fragmentFunction = library.makeFunction(name: "fragmentBody")
        bodyPipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        
        self.nodeRenderPipeline = try! device.makeRenderPipelineState(descriptor: nodePipelineDescriptor)
        
        let pipelineOption : MTLPipelineOption = .failOnBinaryArchiveMiss
        let (bodyRenderPipeline, _) = try! device.makeRenderPipelineState(descriptor: bodyPipelineDescriptor, options: pipelineOption)
        self.bodyRenderPipeline = bodyRenderPipeline
        // Line mesh pipeline (same object/fragment shaders, different mesh function)
        let bodyLinePipelineDescriptor = MTLMeshRenderPipelineDescriptor()
        bodyLinePipelineDescriptor.meshFunction = library.makeFunction(name: "meshLineShader")
        bodyLinePipelineDescriptor.objectFunction = library.makeFunction(name: "objectShader")
        bodyLinePipelineDescriptor.fragmentFunction = library.makeFunction(name: "fragmentBody")
        bodyLinePipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        let (bodyLinePipelineState, _) = try! device.makeRenderPipelineState(descriptor: bodyLinePipelineDescriptor, options: pipelineOption)
        self.bodyLineRenderPipeline = bodyLinePipelineState
        //compute
        
        let initalizeBodiesFunction = library.makeFunction(name: "initalizeBodies")!
        self.initalizeBodiesPipeline = try! device.makeComputePipelineState(function: initalizeBodiesFunction)
        
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
    
    private func initalizeBodies(commandBuffer : borrowing MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        encoder.setComputePipelineState(initalizeBodiesPipeline)
        
        encoder.setBuffer(self.bodyBuffer, offset: 0, index: 0)
        encoder.setBytes(&self.params.physics, length: MemoryLayout<PhysicsParams>.stride, index: 1)

        // Set threadgroup memory for connectionsTile array
        let connectionsTileMemSize = MemoryLayout<SIMD2<UInt32>>.stride * Int(params.blockSize)
        encoder.setThreadgroupMemoryLength(connectionsTileMemSize, index: 0)

        let gridSize = MTLSize(width: (Int(params.numBodies) + threadgroupSize.width - 1) / threadgroupSize.width,
                              height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        
        encoder.endEncoding()
    }
    /*
    private func coalesceConnectionsIndices(commandBuffer: borrowing MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }

        encoder.setComputePipelineState(coalesceConnectionsIndicesPipeline)
        
        encoder.setBuffer(connectionsBuffer, offset: 0, index: 0)
        encoder.setBuffer(perBodyConnectionsBuffer, offset: 0, index: 1)
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
     */
    private func resetTree(commandBuffer: borrowing MTLCommandBuffer) {
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
    
    private func computeBoundingBox(commandBuffer: borrowing MTLCommandBuffer) {
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
    
    private func constructQuadTree(commandBuffer: borrowing MTLCommandBuffer) {
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
    
    private func computeForces(commandBuffer: borrowing MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        encoder.setComputePipelineState(computeForcePipeline)
        encoder.setBuffer(nodeBuffer, offset: 0, index: 0)
        encoder.setBuffer(connectionsBuffer, offset: 0, index: 1)
        encoder.setBuffer(bodyBuffer, offset: 0, index: 2)
        encoder.setBytes(&params.leafLimit, length: MemoryLayout<Int32>.stride, index: 3)
        
        encoder.setBytes(&self.params.physics, length: MemoryLayout<PhysicsParams>.stride, index: 4)
        
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
        
        if !self.bodiesInitialized {
            self.initalizeBodies(commandBuffer: commandBuffer)
            //self.coalesceConnectionsIndices(commandBuffer: commandBuffer)
            self.bodiesInitialized = true
        }
        //self.resetTree(commandBuffer: commandBuffer)
        //self.computeBoundingBox(commandBuffer: commandBuffer)
        //self.constructQuadTree(commandBuffer: commandBuffer)
        self.computeForces(commandBuffer: commandBuffer)
                
        memcpy(self.transformBuffer.contents(), &self.screenTransform, MemoryLayout<ScreenTransform>.stride)

        let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor)!
    
        renderEncoder.setVertexBuffer(self.transformBuffer, offset: 0, index: 1)
        
        // draw bounding box nodes
        renderEncoder.setRenderPipelineState(self.nodeRenderPipeline)
        renderEncoder.setVertexBuffer(self.nodeBuffer, offset: 0, index: 0)
        renderEncoder.drawPrimitives(type: .line, vertexStart: 0, vertexCount: Int(self.params.numNodes) * 8)
        
        // draw bodies 
        // First draw connection lines using the line mesh pipeline (same object data)
        renderEncoder.setRenderPipelineState(self.bodyLineRenderPipeline)
        renderEncoder.setObjectBuffer(self.bodyBuffer, offset: 0, index: 0)
        renderEncoder.setObjectBuffer(self.transformBuffer, offset: 0, index: 1)
        renderEncoder.setObjectBuffer(self.connectionsBuffer, offset: 0, index: 2)

        let gridSize = MTLSize(width: Int(params.numBodies), height: 1, depth: 1)
        let oneThread = MTLSize(width: 1, height: 1, depth: 1)
        renderEncoder.drawMeshThreadgroups(gridSize, threadsPerObjectThreadgroup: oneThread, threadsPerMeshThreadgroup: oneThread)

        // Then draw points on top using the point mesh pipeline (same object data)
        renderEncoder.setRenderPipelineState(self.bodyRenderPipeline)
        renderEncoder.setObjectBuffer(self.bodyBuffer, offset: 0, index: 0)
        renderEncoder.setObjectBuffer(self.transformBuffer, offset: 0, index: 1)
        renderEncoder.setObjectBuffer(self.connectionsBuffer, offset: 0, index: 2)
        renderEncoder.drawMeshThreadgroups(gridSize, threadsPerObjectThreadgroup: oneThread, threadsPerMeshThreadgroup: oneThread)
    
        renderEncoder.endEncoding()
        commandBuffer.present(drawable)
        
        commandBuffer.commit()
        
        //commandBuffer.waitUntilCompleted()
        
        //self.printTreeDebug()
        //self.printPerBodyConnectionIdxDebug()
        //self.debugBodies()
    }
    
    // MARK: - Debug
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
    
    private func printPerBodyConnectionIdxDebug() {
        let uint32Ptr = connectionsBuffer.contents().assumingMemoryBound(to: UInt32.self)
        let perBodyDataSize = 512 + 1  // 512 connection slots + 1 count field
        
        print("\n=== CONNECTIONS DEBUG ===")
        for bodyIdx in 0..<Int(self.params.numBodies) {
            let bodyOffset = bodyIdx * perBodyDataSize
            let numConns = Int(uint32Ptr[bodyOffset + 512])
            
            if numConns == 0 { continue }
            
            print("Body[\(bodyIdx)] has \(numConns) connections:")
            for i in 0..<numConns {
                let connIdx = uint32Ptr[bodyOffset + i]
                print("  -> Body[\(connIdx)]")
            }
        }

    }
    
    private func debugBodies() {
        let bodyPtr = bodyBuffer.contents().assumingMemoryBound(to: Body.self)
        print("\n=== BODIES DEBUG ===")
        for i in 0..<min(10, Int(self.params.numBodies)) {
            let body = bodyPtr[i]
            print("Body[")
            print("\(i)]: Position=(\(body.position.x), \(body.position.y)), Velocity=(\(body.velocity.x), \(body.velocity.y)), Acceleration=(\(body.acceleration.x), \(body.acceleration.y))")
        }
    }
    
    private func debugNodes() {
        let nodePtr = nodeBuffer.contents().assumingMemoryBound(to: Node.self)
        print("\n=== NODES DEBUG ===")
        for i in 0..<min(10, Int(self.params.numNodes)) {
            let node = nodePtr[i]
            print("Node[")
            print("\(i)]: CenterOfMass=(\(node.centerOfMass.x), \(node.centerOfMass.y))")
        }
    }
}
