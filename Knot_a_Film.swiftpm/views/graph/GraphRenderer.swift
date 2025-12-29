import MetalKit
import SharedWithMetal

public final class GraphRenderer : NSObject, MTKViewDelegate {
    
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    
    private var initalizeBodiesPipeline : MTLComputePipelineState
    private var resetPipeline : MTLComputePipelineState
    private var boundingBoxPipeline : MTLComputePipelineState
    private var constructTreePipeline : MTLComputePipelineState
    private var computeForcePipeline : MTLComputePipelineState
    
    private var nodeRenderPipeline: MTLRenderPipelineState
    private var bodyRenderPipeline: MTLRenderPipelineState
    private var bodyLineRenderPipeline: MTLRenderPipelineState
    
    private var nodeTopLeftBuffer: MTLBuffer
    private var nodeBottomRightBuffer: MTLBuffer
    private var nodeCenterOfMassBuffer: MTLBuffer
    private var nodeTotalMassBuffer: MTLBuffer
    private var nodeStartBuffer: MTLBuffer
    private var nodeEndBuffer: MTLBuffer
    private var nodeIsLeafBuffer: MTLBuffer
    
    private var bodyMassBuffer: MTLBuffer
    private var bodyRadiusBuffer: MTLBuffer
    private var bodyPositionBuffer: MTLBuffer
    private var bodyVelocityBuffer: MTLBuffer
    private var bodyAccelerationBuffer: MTLBuffer
    private var bodyInitialIdxBuffer: MTLBuffer
    
    private var bodyMassBufferAlt: MTLBuffer
    private var bodyRadiusBufferAlt: MTLBuffer
    private var bodyPositionBufferAlt: MTLBuffer
    private var bodyVelocityBufferAlt: MTLBuffer
    private var bodyAccelerationBufferAlt: MTLBuffer
    private var bodyInitialIdxBufferAlt: MTLBuffer
    
    private var mutexBuffer : MTLBuffer
    private var transformBuffer : MTLBuffer
    
    public var connectionsBuffer : MTLBuffer
    
    var idxes : [UInt32] = []
    
    private var params : GraphParams
    var bodiesInitialized: Bool = false
    
    public var screenTransform : ScreenTransform = .init(offset: .zero, scale: .init(1.0, 1.0))
    
    private let threadgroupSize = MTLSize(width: 32, height: 1, depth: 1)
    
    
    public init(perBodyConnections : consuming [PerBodyConnectionsData], numConnections : UInt32) {
        
        self.params = GraphParams.makeDefault(numBodies: UInt32(perBodyConnections.count), numConnections: numConnections)
        
        
        let device = MTLCreateSystemDefaultDevice()!
        
        let functionConstants = MTLFunctionConstantValues()
        
        self.params.addMetalFunctionConstants(to: functionConstants)
        
        let numBodies = Int(self.params.numBodies)
        let numNodes = Int(self.params.numNodes)
        
        let maxNodes = Int(MAX_NODES)
        let maxBodies = Int(MAX_BODIES)
        
        self.nodeTopLeftBuffer = device.makeBuffer(length: MemoryLayout<NodeMemberDataFloat2>.stride, options: .storageModeShared)!
        self.nodeBottomRightBuffer = device.makeBuffer(length: MemoryLayout<NodeMemberDataFloat2>.stride, options: .storageModeShared)!
        self.nodeCenterOfMassBuffer = device.makeBuffer(length: MemoryLayout<NodeMemberDataFloat2>.stride, options: .storageModeShared)!
        self.nodeTotalMassBuffer = device.makeBuffer(length: MemoryLayout<NodeMemberDataFloat>.stride, options: .storageModeShared)!
        self.nodeStartBuffer = device.makeBuffer(length: MemoryLayout<NodeMemberDataUInt32>.stride, options: .storageModeShared)!
        self.nodeEndBuffer = device.makeBuffer(length: MemoryLayout<NodeMemberDataUInt32>.stride, options: .storageModeShared)!
        self.nodeIsLeafBuffer = device.makeBuffer(length: MemoryLayout<NodeMemberDataBool>.stride, options: .storageModeShared)!
        
        self.bodyMassBuffer = device.makeBuffer(length: MemoryLayout<BodyMemberDataFloat>.stride, options: .storageModeShared)!
        self.bodyRadiusBuffer = device.makeBuffer(length: MemoryLayout<BodyMemberDataFloat>.stride, options: .storageModeShared)!
        self.bodyPositionBuffer = device.makeBuffer(length: MemoryLayout<BodyMemberDataFloat2>.stride, options: .storageModeShared)!
        self.bodyVelocityBuffer = device.makeBuffer(length: MemoryLayout<BodyMemberDataFloat2>.stride, options: .storageModeShared)!
        self.bodyAccelerationBuffer = device.makeBuffer(length: MemoryLayout<BodyMemberDataFloat2>.stride, options: .storageModeShared)!
        self.bodyInitialIdxBuffer = device.makeBuffer(length: MemoryLayout<BodyMemberDataUInt32>.stride, options: .storageModeShared)!
        
        self.bodyMassBufferAlt = device.makeBuffer(length: MemoryLayout<BodyMemberDataFloat>.stride, options: .storageModeShared)!
        self.bodyRadiusBufferAlt = device.makeBuffer(length: MemoryLayout<BodyMemberDataFloat>.stride, options: .storageModeShared)!
        self.bodyPositionBufferAlt = device.makeBuffer(length: MemoryLayout<BodyMemberDataFloat2>.stride, options: .storageModeShared)!
        self.bodyVelocityBufferAlt = device.makeBuffer(length: MemoryLayout<BodyMemberDataFloat2>.stride, options: .storageModeShared)!
        self.bodyAccelerationBufferAlt = device.makeBuffer(length: MemoryLayout<BodyMemberDataFloat2>.stride, options: .storageModeShared)!
        self.bodyInitialIdxBufferAlt = device.makeBuffer(length: MemoryLayout<BodyMemberDataUInt32>.stride, options: .storageModeShared)!
        
        let n = UInt32(numBodies)
        let nodesN = UInt32(numNodes)

        let numInstancesPtr = bodyMassBuffer.contents().bindMemory(to: BodyMemberDataFloat.self, capacity: 1)
        numInstancesPtr.pointee.numInstances = n

        let numInstancesPtrAlt = bodyMassBufferAlt.contents().bindMemory(to: BodyMemberDataFloat.self, capacity: 1)
        numInstancesPtrAlt.pointee.numInstances = n

        let numRadiusPtr = bodyRadiusBuffer.contents().bindMemory(to: BodyMemberDataFloat.self, capacity: 1)
        numRadiusPtr.pointee.numInstances = n

        let numRadiusPtrAlt = bodyRadiusBufferAlt.contents().bindMemory(to: BodyMemberDataFloat.self, capacity: 1)
        numRadiusPtrAlt.pointee.numInstances = n

        let numPositionPtr = bodyPositionBuffer.contents().bindMemory(to: BodyMemberDataFloat2.self, capacity: 1)
        numPositionPtr.pointee.numInstances = n

        let numPositionPtrAlt = bodyPositionBufferAlt.contents().bindMemory(to: BodyMemberDataFloat2.self, capacity: 1)
        numPositionPtrAlt.pointee.numInstances = n

        let numVelocityPtr = bodyVelocityBuffer.contents().bindMemory(to: BodyMemberDataFloat2.self, capacity: 1)
        numVelocityPtr.pointee.numInstances = n

        let numVelocityPtrAlt = bodyVelocityBufferAlt.contents().bindMemory(to: BodyMemberDataFloat2.self, capacity: 1)
        numVelocityPtrAlt.pointee.numInstances = n

        let numAccelerationPtr = bodyAccelerationBuffer.contents().bindMemory(to: BodyMemberDataFloat2.self, capacity: 1)
        numAccelerationPtr.pointee.numInstances = n

        let numAccelerationPtrAlt = bodyAccelerationBufferAlt.contents().bindMemory(to: BodyMemberDataFloat2.self, capacity: 1)
        numAccelerationPtrAlt.pointee.numInstances = n

        let numInitialIdxPtr = bodyInitialIdxBuffer.contents().bindMemory(to: BodyMemberDataUInt32.self, capacity: 1)
        numInitialIdxPtr.pointee.numInstances = n

        let numInitialIdxPtrAlt = bodyInitialIdxBufferAlt.contents().bindMemory(to: BodyMemberDataUInt32.self, capacity: 1)
        numInitialIdxPtrAlt.pointee.numInstances = n

        let numNodesPtr = nodeTopLeftBuffer.contents().bindMemory(to: NodeMemberDataFloat2.self, capacity: 1)
        numNodesPtr.pointee.numInstances = nodesN

        let numNodesBottomPtr = nodeBottomRightBuffer.contents().bindMemory(to: NodeMemberDataFloat2.self, capacity: 1)
        numNodesBottomPtr.pointee.numInstances = nodesN

        let numCenterPtr = nodeCenterOfMassBuffer.contents().bindMemory(to: NodeMemberDataFloat2.self, capacity: 1)
        numCenterPtr.pointee.numInstances = nodesN

        let numTotalMassPtr = nodeTotalMassBuffer.contents().bindMemory(to: NodeMemberDataFloat.self, capacity: 1)
        numTotalMassPtr.pointee.numInstances = nodesN

        let numStartPtr = nodeStartBuffer.contents().bindMemory(to: NodeMemberDataUInt32.self, capacity: 1)
        numStartPtr.pointee.numInstances = nodesN

        let numEndPtr = nodeEndBuffer.contents().bindMemory(to: NodeMemberDataUInt32.self, capacity: 1)
        numEndPtr.pointee.numInstances = nodesN

        let numIsLeafPtr = nodeIsLeafBuffer.contents().bindMemory(to: NodeMemberDataBool.self, capacity: 1)
        numIsLeafPtr.pointee.numInstances = nodesN
        
        self.connectionsBuffer = device.makeBuffer(length: MemoryLayout<ConnectionsData>.stride, options: .storageModeShared)!
        let connectionsDataPointer = connectionsBuffer.contents().bindMemory(to: ConnectionsData.self, capacity: 1)
        
        withUnsafeMutablePointer(to: &connectionsDataPointer.pointee.connections.0) { connectionsPointer in
            for (i, perBodyConnection) in perBodyConnections.enumerated() {
                connectionsPointer.advanced(by: i).pointee = perBodyConnection
            }
        }
        
        connectionsDataPointer.pointee.numConnections = uint(numConnections)
        print("Initialized \(numConnections) connections")
        
        let mutexSize = MemoryLayout<Int32>.stride * Int(self.params.numNodes)
        self.mutexBuffer = device.makeBuffer(length: mutexSize, options: .storageModePrivate)!
        
        let transformSize = MemoryLayout<ScreenTransform>.size
        self.transformBuffer = device.makeBuffer(length: transformSize,
                                                 options: .cpuCacheModeWriteCombined)!
        
        let shadersBundleURL = Bundle.main.bundleURL.appendingPathComponent("../Knot a Film_MetalShaders.bundle")
        print(shadersBundleURL)
        let bundle = Bundle(url: shadersBundleURL)!
        let libraryURL = bundle.url(forResource: "debug", withExtension: "metallib")!
        //let library = try! device.makeLibrary(URL: libraryURL)
        
        let library = try! device.makeDefaultLibrary(bundle: .main)

        
        print(library)
        
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
        
        let bodyLinePipelineDescriptor = MTLMeshRenderPipelineDescriptor()
        bodyLinePipelineDescriptor.meshFunction = library.makeFunction(name: "meshLineShader")
        bodyLinePipelineDescriptor.objectFunction = library.makeFunction(name: "objectShader")
        bodyLinePipelineDescriptor.fragmentFunction = library.makeFunction(name: "fragmentBody")
        bodyLinePipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        let (bodyLinePipelineState, _) = try! device.makeRenderPipelineState(descriptor: bodyLinePipelineDescriptor, options: pipelineOption)
        self.bodyLineRenderPipeline = bodyLinePipelineState
        
        let initalizeBodiesFunction = try! library.makeFunction(name: "initalizeBodies", constantValues: functionConstants)
        self.initalizeBodiesPipeline = try! device.makeComputePipelineState(function: initalizeBodiesFunction)
        
        let resetFunction = try! library.makeFunction(name: "resetKernel", constantValues: functionConstants)
        self.resetPipeline = try! device.makeComputePipelineState(function: resetFunction)
        
        let boundingBoxFunction = try! library.makeFunction(name: "computeBoundingBoxKernel", constantValues: functionConstants)
        self.boundingBoxPipeline = try! device.makeComputePipelineState(function: boundingBoxFunction)
        
        let computeForceKernel = try! library.makeFunction(name: "computeForceKernel", constantValues: functionConstants)
        self.computeForcePipeline = try! device.makeComputePipelineState(function: computeForceKernel)
        
        let constructTreeFunction = try! library.makeFunction(name: "constructQuadTreeKernel", constantValues: functionConstants)
        self.constructTreePipeline = try! device.makeComputePipelineState(function: constructTreeFunction)
        
        self.commandQueue = device.makeCommandQueue()!
        
        self.device = device
        super.init()
    }
    
    private func initalizeBodies(commandBuffer : borrowing MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        encoder.setComputePipelineState(initalizeBodiesPipeline)
        
        encoder.setBuffer(self.bodyMassBuffer, offset: 0, index: Int(BODY_MASS_IDX))
        encoder.setBuffer(self.bodyRadiusBuffer, offset: 0, index: Int(BODY_RADIUS_IDX))
        encoder.setBuffer(self.bodyPositionBuffer, offset: 0, index: Int(BODY_POSITION_IDX))
        encoder.setBuffer(self.bodyVelocityBuffer, offset: 0, index: Int(BODY_VELOCITY_IDX))
        encoder.setBuffer(self.bodyAccelerationBuffer, offset: 0, index: Int(BODY_ACCELERATION_IDX))
        encoder.setBuffer(self.bodyInitialIdxBuffer, offset: 0, index: Int(BODY_INITIAL_IDX_IDX))
        encoder.setBytes(&self.params.physics, length: MemoryLayout<PhysicsParams>.stride, index: 22)
        
        let gridSize = MTLSize(width: (Int(params.numBodies) + threadgroupSize.width - 1) / threadgroupSize.width,
                               height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        
        encoder.endEncoding()
    }
    
    private func resetTree(commandBuffer: borrowing MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        encoder.setComputePipelineState(resetPipeline)
        encoder.setBuffer(nodeTopLeftBuffer, offset: 0, index: Int(NODE_TOP_LEFT_IDX))
        encoder.setBuffer(nodeBottomRightBuffer, offset: 0, index: Int(NODE_BOTTOM_RIGHT_IDX))
        encoder.setBuffer(nodeCenterOfMassBuffer, offset: 0, index: Int(NODE_CENTER_OF_MASS_IDX))
        encoder.setBuffer(nodeTotalMassBuffer, offset: 0, index: Int(NODE_TOTAL_MASS_IDX))
        encoder.setBuffer(nodeStartBuffer, offset: 0, index: Int(NODE_START_IDX))
        encoder.setBuffer(nodeEndBuffer, offset: 0, index: Int(NODE_END_IDX))
        encoder.setBuffer(nodeIsLeafBuffer, offset: 0, index: Int(NODE_IS_LEAF_IDX))
        encoder.setBuffer(mutexBuffer, offset: 0, index: Int(MUTEX_IDX))
        
        let gridSize = MTLSize(width: (Int(params.numNodes) + threadgroupSize.width - 1) / threadgroupSize.width,
                               height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
    
    private func computeBoundingBox(commandBuffer: borrowing MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        encoder.setComputePipelineState(boundingBoxPipeline)
        encoder.setBuffer(nodeTopLeftBuffer, offset: 0, index: Int(NODE_TOP_LEFT_IDX))
        encoder.setBuffer(nodeBottomRightBuffer, offset: 0, index: Int(NODE_BOTTOM_RIGHT_IDX))
        encoder.setBuffer(bodyPositionBuffer, offset: 0, index: Int(BODY_POSITION_IDX))
        encoder.setBuffer(mutexBuffer, offset: 0, index: Int(MUTEX_IDX))
        
        let gridSize = MTLSize(width: (Int(params.numBodies) + threadgroupSize.width - 1) / threadgroupSize.width,
                               height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
    
    private func constructQuadTree(commandBuffer: borrowing MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        encoder.setComputePipelineState(constructTreePipeline)
        
        var currentMass = bodyMassBuffer
        var currentRadius = bodyRadiusBuffer
        var currentPosition = bodyPositionBuffer
        var currentVelocity = bodyVelocityBuffer
        var currentAcceleration = bodyAccelerationBuffer
        var currentInitialIdx = bodyInitialIdxBuffer
        
        var nextMass = bodyMassBufferAlt
        var nextRadius = bodyRadiusBufferAlt
        var nextPosition = bodyPositionBufferAlt
        var nextVelocity = bodyVelocityBufferAlt
        var nextAcceleration = bodyAccelerationBufferAlt
        var nextInitialIdx = bodyInitialIdxBufferAlt
        
        encoder.setBuffer(nodeTopLeftBuffer, offset: 0, index: Int(NODE_TOP_LEFT_IDX))
        encoder.setBuffer(nodeBottomRightBuffer, offset: 0, index: Int(NODE_BOTTOM_RIGHT_IDX))
        encoder.setBuffer(nodeCenterOfMassBuffer, offset: 0, index: Int(NODE_CENTER_OF_MASS_IDX))
        encoder.setBuffer(nodeTotalMassBuffer, offset: 0, index: Int(NODE_TOTAL_MASS_IDX))
        encoder.setBuffer(nodeStartBuffer, offset: 0, index: Int(NODE_START_IDX))
        encoder.setBuffer(nodeEndBuffer, offset: 0, index: Int(NODE_END_IDX))
        encoder.setBuffer(nodeIsLeafBuffer, offset: 0, index: Int(NODE_IS_LEAF_IDX))
        
        for level in 0...self.params.maxDepth {
            let nodesInLevel = Int(pow(4.0, Double(level)))
            var nodeOffset = (Int32(pow(4.0, Double(level))) - 1) / 3
            
            encoder.setBuffer(currentMass, offset: 0, index: Int(BODY_MASS_IDX))
            encoder.setBuffer(currentRadius, offset: 0, index: Int(BODY_RADIUS_IDX))
            encoder.setBuffer(currentPosition, offset: 0, index: Int(BODY_POSITION_IDX))
            encoder.setBuffer(currentVelocity, offset: 0, index: Int(BODY_VELOCITY_IDX))
            encoder.setBuffer(currentAcceleration, offset: 0, index: Int(BODY_ACCELERATION_IDX))
            encoder.setBuffer(currentInitialIdx, offset: 0, index: Int(BODY_INITIAL_IDX_IDX))
            
            encoder.setBuffer(nextMass, offset: 0, index: Int(BODY_MASS_ALT_IDX))
            encoder.setBuffer(nextRadius, offset: 0, index: Int(BODY_RADIUS_ALT_IDX))
            encoder.setBuffer(nextPosition, offset: 0, index: Int(BODY_POSITION_ALT_IDX))
            encoder.setBuffer(nextVelocity, offset: 0, index: Int(BODY_VELOCITY_ALT_IDX))
            encoder.setBuffer(nextAcceleration, offset: 0, index: Int(BODY_ACCELERATION_ALT_IDX))
            encoder.setBuffer(nextInitialIdx, offset: 0, index: Int(BODY_INITIAL_IDX_ALT_IDX))
            
            encoder.setBytes(&nodeOffset, length: MemoryLayout<Int32>.stride, index: 22)
            
            let countMemSize = MemoryLayout<Int32>.stride * 8
            encoder.setThreadgroupMemoryLength(countMemSize, index: 0)
            
            let gridSize = MTLSize(width: nodesInLevel, height: 1, depth: 1)
            encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
            
            swap(&currentMass, &nextMass)
            swap(&currentRadius, &nextRadius)
            swap(&currentPosition, &nextPosition)
            swap(&currentVelocity, &nextVelocity)
            swap(&currentAcceleration, &nextAcceleration)
            swap(&currentInitialIdx, &nextInitialIdx)
        }
        
        encoder.endEncoding()
        
        if self.params.maxDepth % 2 == 1 {
            swap(&currentMass, &nextMass)
            swap(&currentRadius, &nextRadius)
            swap(&currentPosition, &nextPosition)
            swap(&currentVelocity, &nextVelocity)
            swap(&currentAcceleration, &nextAcceleration)
            swap(&currentInitialIdx, &nextInitialIdx)
        }
    }
    
    private func computeForces(commandBuffer: borrowing MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        encoder.setComputePipelineState(computeForcePipeline)
        
        encoder.setBuffer(nodeTopLeftBuffer, offset: 0, index: Int(NODE_TOP_LEFT_IDX))
        encoder.setBuffer(nodeBottomRightBuffer, offset: 0, index: Int(NODE_BOTTOM_RIGHT_IDX))
        encoder.setBuffer(nodeCenterOfMassBuffer, offset: 0, index: Int(NODE_CENTER_OF_MASS_IDX))
        encoder.setBuffer(nodeIsLeafBuffer, offset: 0, index: Int(NODE_IS_LEAF_IDX))
        encoder.setBuffer(connectionsBuffer, offset: 0, index: Int(CONNECTIONS_IDX))
        encoder.setBuffer(bodyMassBuffer, offset: 0, index: Int(BODY_MASS_IDX))
        encoder.setBuffer(bodyRadiusBuffer, offset: 0, index: Int(BODY_RADIUS_IDX))
        encoder.setBuffer(bodyPositionBuffer, offset: 0, index: Int(BODY_POSITION_IDX))
        encoder.setBuffer(bodyVelocityBuffer, offset: 0, index: Int(BODY_VELOCITY_IDX))
        encoder.setBuffer(bodyAccelerationBuffer, offset: 0, index: Int(BODY_ACCELERATION_IDX))
        encoder.setBuffer(bodyInitialIdxBuffer, offset: 0, index: Int(BODY_INITIAL_IDX_IDX))
        encoder.setBytes(&self.params.physics, length: MemoryLayout<PhysicsParams>.stride, index: Int(PHYSICS_PARAMS_IDX))
        
        let gridSize = MTLSize(width: (Int(params.numBodies) + threadgroupSize.width - 1) / threadgroupSize.width,
                               height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
    
    public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) { }
    
    public func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable, let descriptor = view.currentRenderPassDescriptor else {
            return
        }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        
        defer { self.bodiesInitialized = true }
        if !self.bodiesInitialized {
            self.initalizeBodies(commandBuffer: commandBuffer)
        }
        
        if self.params.useBarnes {
            self.resetTree(commandBuffer: commandBuffer)
            self.computeBoundingBox(commandBuffer: commandBuffer)
            self.constructQuadTree(commandBuffer: commandBuffer)
        }
        
        self.computeForces(commandBuffer: commandBuffer)
        
        memcpy(self.transformBuffer.contents(), &self.screenTransform, MemoryLayout<ScreenTransform>.stride)
        
        let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor)!
        
        renderEncoder.setVertexBuffer(self.transformBuffer, offset: 0, index: Int(SCREEN_TRANSFORM_IDX))
        
        renderEncoder.setRenderPipelineState(self.nodeRenderPipeline)
        renderEncoder.setVertexBuffer(nodeTopLeftBuffer, offset: 0, index: Int(NODE_TOP_LEFT_IDX))
        renderEncoder.setVertexBuffer(nodeBottomRightBuffer, offset: 0, index: Int(NODE_BOTTOM_RIGHT_IDX))
        
        renderEncoder.drawPrimitives(type: .line, vertexStart: 0, vertexCount: Int(self.params.numNodes) * 8)
        
        renderEncoder.setRenderPipelineState(self.bodyLineRenderPipeline)
        renderEncoder.setObjectBuffer(self.bodyPositionBuffer, offset: 0, index: Int(BODY_POSITION_IDX))
        renderEncoder.setObjectBuffer(self.transformBuffer, offset: 0, index: Int(SCREEN_TRANSFORM_IDX))
        renderEncoder.setObjectBuffer(self.connectionsBuffer, offset: 0, index: Int(CONNECTIONS_IDX))
        
        let gridSize = MTLSize(width: Int(params.numBodies), height: 1, depth: 1)
        let oneThread = MTLSize(width: 1, height: 1, depth: 1)
        renderEncoder.drawMeshThreadgroups(gridSize, threadsPerObjectThreadgroup: oneThread, threadsPerMeshThreadgroup: oneThread)
        
        renderEncoder.setRenderPipelineState(self.bodyRenderPipeline)
        renderEncoder.drawMeshThreadgroups(gridSize, threadsPerObjectThreadgroup: oneThread, threadsPerMeshThreadgroup: oneThread)
        
        renderEncoder.endEncoding()
        commandBuffer.present(drawable)
        
        commandBuffer.commit()
        
        commandBuffer.waitUntilCompleted()
        //self.printPerBodyConnectionIdx()
        self.printBodies()
        //self.printTree()
        
    }
    
    // MARK: - Debug
    private func printTree() {
        let topLeftPtr = nodeTopLeftBuffer.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        let bottomRightPtr = nodeBottomRightBuffer.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        let centerOfMassPtr = nodeCenterOfMassBuffer.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        let totalMassPtr = nodeTotalMassBuffer.contents().assumingMemoryBound(to: Float.self)
        let startPtr = nodeStartBuffer.contents().assumingMemoryBound(to: UInt32.self)
        let endPtr = nodeEndBuffer.contents().assumingMemoryBound(to: UInt32.self)
        let isLeafPtr = nodeIsLeafBuffer.contents().assumingMemoryBound(to: Bool.self)
        
        var leafCount = 0, internalCount = 0, emptyCount = 0
        
        print("\n=== TREE STRUCTURE DEBUG ===")
        for i in 0..<min(21, Int(self.params.numNodes)) {
            let start = startPtr[i]
            let end = endPtr[i]
            let isLeaf = isLeafPtr[i]
            let isEmpty = start == UInt32.max
            let statusStr = isEmpty ? "EMPTY" : (isLeaf ? "LEAF" : "INTERNAL")
            let mass = totalMassPtr[i]
            let com = centerOfMassPtr[i]
            let topLeft = topLeftPtr[i]
            let bottomRight = bottomRightPtr[i]
            
            print("Node[\(i)]: \(statusStr), start=\(start), end=\(end), mass=\(mass), COM=(\(com.x), \(com.y))")
            print("         bounds: topLeft=(\(topLeft.x), \(topLeft.y)), bottomRight=(\(bottomRight.x), \(bottomRight.y))")
            
            if isEmpty {
                emptyCount += 1
            } else if isLeaf {
                leafCount += 1
            } else {
                internalCount += 1
            }
        }
        
        for i in 21..<Int(self.params.numNodes) {
            let start = startPtr[i]
            let isLeaf = isLeafPtr[i]
            if start == UInt32.max { emptyCount += 1 }
            else if isLeaf { leafCount += 1 }
            else { internalCount += 1 }
        }
        
        print("TOTAL: \(leafCount) leaves, \(internalCount) internal, \(emptyCount) empty")
    }
    
    private func printPerBodyConnectionIdx() {
        let uint32Ptr = connectionsBuffer.contents().assumingMemoryBound(to: UInt32.self)
        let perBodyDataSize = 512 + 1
        
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
    
    private func printBodies() {
        let massPtr = bodyMassBuffer.contents().assumingMemoryBound(to: Float.self)
        let radiusPtr = bodyRadiusBuffer.contents().assumingMemoryBound(to: Float.self)
        let positionPtr = bodyPositionBuffer.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        let velocityPtr = bodyVelocityBuffer.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        let accelerationPtr = bodyAccelerationBuffer.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        let initialIdxPtr = bodyInitialIdxBuffer.contents().assumingMemoryBound(to: UInt32.self)
        
        print("\n=== BODIES DEBUG ===")
        
        for i in 0..<Int(self.params.numBodies) {
            let position = positionPtr[i]
            let velocity = velocityPtr[i]
            let acceleration = accelerationPtr[i]
            let initialIdx = initialIdxPtr[i]
            
            if !self.bodiesInitialized {
                self.idxes.append(initialIdx)
            } else if self.bodiesInitialized && initialIdx != self.idxes[i] {
                print("error")
            }
            
            print("\(i)]: Position=(\(position.x), \(position.y)), Velocity=(\(velocity.x), \(velocity.y)), Acceleration=(\(acceleration.x), \(acceleration.y)), InitialIdx = \(initialIdx)")
        }
    }
    
    private func printNodes() {
        let centerOfMassPtr = nodeCenterOfMassBuffer.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        
        print("\n=== NODES DEBUG ===")
        for i in 0..<min(10, Int(self.params.numNodes)) {
            let com = centerOfMassPtr[i]
            print("Node[\(i)]: CenterOfMass=(\(com.x), \(com.y))")
        }
    }
}
