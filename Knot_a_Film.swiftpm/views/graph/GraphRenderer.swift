import MetalKit
import SharedWithMetal

protocol MTLBufferGroup {
    static func makeBuffers(numInstances: UInt32, device: MTLDevice, options: MTLResourceOptions) -> Self
}

public struct NodeBufferGroup : MTLBufferGroup {
    public var topLeftBuffer: MTLBuffer
    public var bottomRightBuffer: MTLBuffer
    public var centerOfMassBuffer: MTLBuffer
    public var totalMassBuffer: MTLBuffer
    public var startBuffer: MTLBuffer
    public var endBuffer: MTLBuffer
    public var isLeafBuffer: MTLBuffer
    
    static func makeBuffers(numInstances: UInt32, device: MTLDevice, options: MTLResourceOptions) -> Self {
        let topLeftBuffer = device.makeBuffer(length: MemoryLayout<NodeMemberDataFloat2>.stride, options: options)!
        let bottomRightBuffer = device.makeBuffer(length: MemoryLayout<NodeMemberDataFloat2>.stride, options: options)!
        let centerOfMassBuffer = device.makeBuffer(length: MemoryLayout<NodeMemberDataFloat2>.stride, options: options)!
        let totalMassBuffer = device.makeBuffer(length: MemoryLayout<NodeMemberDataFloat>.stride, options: options)!
        let startBuffer = device.makeBuffer(length: MemoryLayout<NodeMemberDataUInt32>.stride, options: options)!
        let endBuffer = device.makeBuffer(length: MemoryLayout<NodeMemberDataUInt32>.stride, options: options)!
        let isLeafBuffer = device.makeBuffer(length: MemoryLayout<NodeMemberDataBool>.stride, options: options)!
        
        let numNodesPtr = topLeftBuffer.contents().bindMemory(to: NodeMemberDataFloat2.self, capacity: 1)
        numNodesPtr.pointee.numNodes = numInstances

        let numNodesBottomPtr = bottomRightBuffer.contents().bindMemory(to: NodeMemberDataFloat2.self, capacity: 1)
        numNodesBottomPtr.pointee.numNodes = numInstances

        let numCenterPtr = centerOfMassBuffer.contents().bindMemory(to: NodeMemberDataFloat2.self, capacity: 1)
        numCenterPtr.pointee.numNodes = numInstances

        let numTotalMassPtr = totalMassBuffer.contents().bindMemory(to: NodeMemberDataFloat.self, capacity: 1)
        numTotalMassPtr.pointee.numNodes = numInstances

        let numStartPtr = startBuffer.contents().bindMemory(to: NodeMemberDataUInt32.self, capacity: 1)
        numStartPtr.pointee.numNodes = numInstances

        let numEndPtr = endBuffer.contents().bindMemory(to: NodeMemberDataUInt32.self, capacity: 1)
        numEndPtr.pointee.numNodes = numInstances

        let numIsLeafPtr = isLeafBuffer.contents().bindMemory(to: NodeMemberDataBool.self, capacity: 1)
        numIsLeafPtr.pointee.numNodes = numInstances
        
        return .init(topLeftBuffer: consume topLeftBuffer,
                     bottomRightBuffer: consume bottomRightBuffer,
                     centerOfMassBuffer: consume centerOfMassBuffer,
                     totalMassBuffer: consume totalMassBuffer,
                     startBuffer: consume startBuffer,
                     endBuffer: consume endBuffer,
                     isLeafBuffer: consume isLeafBuffer)
    }
}

public struct BodyBufferGroup : MTLBufferGroup {
    public var massBuffer: MTLBuffer
    public var radiusBuffer: MTLBuffer
    public var positionBuffer: MTLBuffer
    public var velocityBuffer: MTLBuffer
    public var accelerationBuffer: MTLBuffer
    public var initialIdxBuffer: MTLBuffer
    public var offsetsBuffer: MTLBuffer
    
    public var massAltBuffer: MTLBuffer
    public var positionAltBuffer: MTLBuffer
    public var initialIdxAltBuffer: MTLBuffer
    
    static func makeBuffers(numInstances: UInt32, device: MTLDevice, options: MTLResourceOptions) -> Self {
        let massBuffer = device.makeBuffer(length: MemoryLayout<BodyMemberDataFloat>.stride, options: .storageModeShared)!
        let radiusBuffer = device.makeBuffer(length: MemoryLayout<BodyMemberDataFloat>.stride, options: .storageModeShared)!
        let positionBuffer = device.makeBuffer(length: MemoryLayout<BodyMemberDataFloat2>.stride, options: .storageModeShared)!
        let velocityBuffer = device.makeBuffer(length: MemoryLayout<BodyMemberDataFloat2>.stride, options: .storageModeShared)!
        let accelerationBuffer = device.makeBuffer(length: MemoryLayout<BodyMemberDataFloat2>.stride, options: .storageModeShared)!
        let initialIdxBuffer = device.makeBuffer(length: MemoryLayout<BodyMemberDataUInt32>.stride, options: .storageModeShared)!
        let offsetsBuffer = device.makeBuffer(length: MemoryLayout<BodyMemberDataUInt32>.stride, options: .storageModeShared)!
        
        let massAltBuffer = device.makeBuffer(length: MemoryLayout<BodyMemberDataFloat>.stride, options: .storageModeShared)!
        let positionAltBuffer = device.makeBuffer(length: MemoryLayout<BodyMemberDataFloat2>.stride, options: .storageModeShared)!
        let initialIdxAltBuffer = device.makeBuffer(length: MemoryLayout<BodyMemberDataUInt32>.stride, options: .storageModeShared)!
        
        let numInstancesPtr = massBuffer.contents().bindMemory(to: BodyMemberDataFloat.self, capacity: 1)
        numInstancesPtr.pointee.numBodies = numInstances

        let numRadiusPtr = radiusBuffer.contents().bindMemory(to: BodyMemberDataFloat.self, capacity: 1)
        numRadiusPtr.pointee.numBodies = numInstances

        let numPositionPtr = positionBuffer.contents().bindMemory(to: BodyMemberDataFloat2.self, capacity: 1)
        numPositionPtr.pointee.numBodies = numInstances

        let numVelocityPtr = velocityBuffer.contents().bindMemory(to: BodyMemberDataFloat2.self, capacity: 1)
        numVelocityPtr.pointee.numBodies = numInstances

        let numAccelerationPtr = accelerationBuffer.contents().bindMemory(to: BodyMemberDataFloat2.self, capacity: 1)
        numAccelerationPtr.pointee.numBodies = numInstances

        let numInitialIdxPtr = initialIdxBuffer.contents().bindMemory(to: BodyMemberDataUInt32.self, capacity: 1)
        numInitialIdxPtr.pointee.numBodies = numInstances
        
        let numOffsetsPtr = offsetsBuffer.contents().bindMemory(to: BodyMemberDataUInt32.self, capacity: 1)
        numOffsetsPtr.pointee.numBodies = numInstances
        
        let massAltPtr = massAltBuffer.contents().bindMemory(to: BodyMemberDataFloat.self, capacity: 1)
        massAltPtr.pointee.numBodies = numInstances

        let positionAltPtr = positionAltBuffer.contents().bindMemory(to: BodyMemberDataFloat2.self, capacity: 1)
        positionAltPtr.pointee.numBodies = numInstances

        let initialIdxAltPtr = initialIdxAltBuffer.contents().bindMemory(to: BodyMemberDataUInt32.self, capacity: 1)
        initialIdxAltPtr.pointee.numBodies = numInstances
        
        return BodyBufferGroup(massBuffer: consume massBuffer,
                               radiusBuffer: consume radiusBuffer,
                               positionBuffer: consume positionBuffer,
                               velocityBuffer: consume velocityBuffer,
                               accelerationBuffer: consume accelerationBuffer,
                               initialIdxBuffer: consume initialIdxBuffer,
                               offsetsBuffer: consume offsetsBuffer,
                               massAltBuffer: consume massAltBuffer,
                               positionAltBuffer: consume positionAltBuffer,
                               initialIdxAltBuffer: consume initialIdxAltBuffer)
    }
    
    public func setSentinal(_ numBodies : UInt32, _ numEdges : UInt32) {
        // Write sentinel at offsets[numBodies] = total number of flattened edges so mesh shaders
        // can safely read offsets[gid+1]. Also perform basic bounds checks against Metal header limits.
        let numOffsetsPtr = offsetsBuffer.contents().bindMemory(to: BodyMemberDataUInt32.self, capacity: 1)
        withUnsafeMutablePointer(to: &numOffsetsPtr.pointee.data.0) { offsetsPtr in
            offsetsPtr.advanced(by: Int(numBodies)).pointee = numEdges
        }
    }
}

public struct EdgesBufferGroup : MTLBufferGroup {
    public var edgeIndiciesBuffer : MTLBuffer
    public var sortedEdgeIndiciesBuffer : MTLBuffer
    public var bodyIndiciesBuffer : MTLBuffer
    
    static func makeBuffers(numInstances: UInt32, device: MTLDevice, options: MTLResourceOptions) -> Self {
        let edgeIndiciesBuffer = device.makeBuffer(length: MemoryLayout<EdgesMemberDataUInt32>.stride, options: .storageModeShared)!
        let bodyIndiciesBuffer = device.makeBuffer(length: MemoryLayout<EdgesMemberDataUInt32>.stride, options: .storageModeShared)!
        let sortedEdgeIndiciesBuffer = device.makeBuffer(length: MemoryLayout<EdgesMemberDataUInt32>.stride, options: .storageModeShared)!

        let bodyIndiciesPointer = bodyIndiciesBuffer.contents().bindMemory(to: EdgesMemberDataUInt32.self, capacity: 1)
        bodyIndiciesPointer.pointee.numBodies = numInstances
        
        let edgeIndiciesPointer = edgeIndiciesBuffer.contents().bindMemory(to: EdgesMemberDataUInt32.self, capacity: 1)
        edgeIndiciesPointer.pointee.numBodies = numInstances
        
        let sortedEdgeIndiciesPointer = sortedEdgeIndiciesBuffer.contents().bindMemory(to: EdgesMemberDataUInt32.self, capacity: 1)
        sortedEdgeIndiciesPointer.pointee.numBodies = numInstances
        
        return EdgesBufferGroup(edgeIndiciesBuffer: consume edgeIndiciesBuffer,
                                sortedEdgeIndiciesBuffer: consume sortedEdgeIndiciesBuffer,
                                bodyIndiciesBuffer: consume bodyIndiciesBuffer)
    }

}

public final class GraphRenderer : NSObject, MTKViewDelegate {
    
    let device : MTLDevice
    let commandQueue : MTLCommandQueue
    
    private var initalizeEdgesPipeline : MTLComputePipelineState
    private var initalizeBodiesPipeline : MTLComputePipelineState
    private var resetPipeline : MTLComputePipelineState
    private var boundingBoxPipeline : MTLComputePipelineState
    private var constructTreePipeline : MTLComputePipelineState
    private var computeForcePipeline : MTLComputePipelineState
    
    private var nodeRenderPipeline : MTLRenderPipelineState
    private var bodyRenderPipeline : MTLRenderPipelineState
    private var bodyLineRenderPipeline : MTLRenderPipelineState
    
    private var nodeData : NodeBufferGroup
    private var bodyData : BodyBufferGroup
    private var edgesData : EdgesBufferGroup
    
    private var mutexBuffer : MTLBuffer
    private var transformBuffer : MTLBuffer
    
    public var edgesBuffer : MTLBuffer
    public var offsetsBuffer : MTLBuffer
            
    private var params : GraphParams
    
    var bodiesInitialized : Bool = false
    let benchmarkOnly : Bool = false
    let counterSampleBuffer : MTLCounterSampleBuffer
    var sampleIndex = 0

    
    public var screenTransform : ScreenTransform = .init(offset: .zero, scale: .init(1.0, 1.0))
    
    private let threadgroupSize = MTLSize(width: 32, height: 1, depth: 1)
    
    public init(edges: consuming [UInt32], offsets: consuming [UInt32], numEdges: UInt32) {
        
        self.params = GraphParams.makeDefault(numBodies: UInt32(offsets.count), numEdges: numEdges)
        
        let device = MTLCreateSystemDefaultDevice()!
        
        let functionConstants = MTLFunctionConstantValues()
        
        self.params.addMetalFunctionConstants(to: functionConstants)
        
        //create bodies
        let numBodies = UInt32(self.params.numBodies)
        self.bodyData = BodyBufferGroup.makeBuffers(numInstances: numBodies, device: device, options: .storageModeShared)
        self.bodyData.setSentinal(numBodies, numEdges)
        
        //create nodes
        let numNodes = UInt32(self.params.numNodes)
        self.nodeData = NodeBufferGroup.makeBuffers(numInstances: numNodes, device: device, options: .storageModeShared)
        
        let numEdges = UInt32(offsets.count)
        self.edgesData = EdgesBufferGroup.makeBuffers(numInstances: numEdges, device: device, options: .storageModeShared)

        #if DEBUG
        if Int(numEdges) > MAX_EDGES * MAX_BODIES {
            print("Warning: total edges (\(numEdges)) exceed MAX_EDGES (\(MAX_EDGES * MAX_BODIES)). Clamping may be required.")
        }
        if offsets.count > MAX_BODIES {
            print("Warning: number of bodies (\(offsets.count)) exceed MAX_BODIES (\(MAX_BODIES)).")
        }
        #endif

        print("Initialized \(offsets.count) bodies with edges (\(numEdges) total edges)")
        
        self.edgesBuffer = device.makeBuffer(bytes: edges, length: MemoryLayout<UInt32>.stride * edges.count, options: .storageModeShared)!
        self.offsetsBuffer = device.makeBuffer(bytes: offsets, length: MemoryLayout<UInt32>.stride * offsets.count, options: .storageModeShared)!


        let mutexSize = MemoryLayout<Int32>.stride * Int(self.params.numNodes)
        self.mutexBuffer = device.makeBuffer(length: mutexSize, options: .storageModePrivate)!
        
        let transformSize = MemoryLayout<ScreenTransform>.size
        self.transformBuffer = device.makeBuffer(length: transformSize,
                                                 options: .cpuCacheModeWriteCombined)!
        
        let shadersBundleURL = Bundle.main.bundleURL.appendingPathComponent("../Knot a Film_MetalShaders.bundle")
        let bundle = Bundle(url: shadersBundleURL)!
        let libraryURL = bundle.url(forResource: "debug", withExtension: "metallib")!
        let library = try! device.makeLibrary(URL: libraryURL)
        
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
        
        let initalizeEdgesFunction = try! library.makeFunction(name: "initalizeEdges", constantValues: functionConstants)
        self.initalizeEdgesPipeline = try! device.makeComputePipelineState(function: initalizeEdgesFunction)
        
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
        
        let counterSet = device.counterSets?.first {
            $0.name == MTLCommonCounterSet.timestamp.rawValue
        }

        let sampleCount = 16  // enough for all stages
        let descriptor = MTLCounterSampleBufferDescriptor()
        descriptor.counterSet = counterSet
        descriptor.storageMode = .shared
        descriptor.sampleCount = sampleCount

        self.counterSampleBuffer = try! device.makeCounterSampleBuffer(descriptor: descriptor)

        
        self.device = device
        super.init()
    }
    
    // MARK: - Kernels
    private func initalizeEdges(commandBuffer: borrowing MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        encoder.setComputePipelineState(initalizeEdgesPipeline)
        
        encoder.setBuffer(self.edgesBuffer, offset: 0, index: 0)
        encoder.setBuffer(self.offsetsBuffer, offset: 0, index: 1)
        encoder.setBuffer(self.bodyData.offsetsBuffer, offset: 0, index: Int(BODY_OFFSETS_IDX))
        encoder.setBuffer(self.edgesData.edgeIndiciesBuffer, offset: 0, index: Int(EDGE_INDICIES_IDX))
        encoder.setBuffer(self.edgesData.bodyIndiciesBuffer, offset: 0, index: Int(EDGE_BODY_IDX))

        let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
        let elementsPerThread = 4
        
        encoder.setThreadgroupMemoryLength(
            MemoryLayout<Int32>.stride * threadgroupSize.width * elementsPerThread,
            index: 0
        )
        encoder.setThreadgroupMemoryLength(
            MemoryLayout<Int32>.stride * (threadgroupSize.width / 32),
            index: 1
        )

        let totalElements = Int(params.numEdges)
        let elementsPerThreadgroup = threadgroupSize.width * elementsPerThread
        let gridSize = MTLSize(
            width: (totalElements + elementsPerThreadgroup - 1) / elementsPerThreadgroup,
            height: 1,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
    
    private func initalizeBodies(commandBuffer : borrowing MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        encoder.setComputePipelineState(initalizeBodiesPipeline)
        
        encoder.setBuffer(self.bodyData.massBuffer, offset: 0, index: Int(BODY_MASS_IDX))
        encoder.setBuffer(self.bodyData.radiusBuffer, offset: 0, index: Int(BODY_RADIUS_IDX))
        encoder.setBuffer(self.bodyData.positionBuffer, offset: 0, index: Int(BODY_POSITION_IDX))
        encoder.setBuffer(self.bodyData.velocityBuffer, offset: 0, index: Int(BODY_VELOCITY_IDX))
        encoder.setBuffer(self.bodyData.accelerationBuffer, offset: 0, index: Int(BODY_ACCELERATION_IDX))
        encoder.setBuffer(self.bodyData.initialIdxBuffer, offset: 0, index: Int(BODY_INITIAL_IDX_IDX))
        encoder.setBytes(&params.physics, length: MemoryLayout<PhysicsParams>.stride, index: Int(PHYSICS_PARAMS_IDX))
        
        let gridSize = MTLSize(width: (Int(params.numBodies) + threadgroupSize.width - 1) / threadgroupSize.width,
                               height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        
        encoder.endEncoding()
    }
    
    private func resetTree(commandBuffer: borrowing MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        encoder.setComputePipelineState(resetPipeline)
        encoder.setBuffer(self.nodeData.topLeftBuffer, offset: 0, index: Int(NODE_TOP_LEFT_IDX))
        encoder.setBuffer(self.nodeData.bottomRightBuffer, offset: 0, index: Int(NODE_BOTTOM_RIGHT_IDX))
        encoder.setBuffer(self.nodeData.centerOfMassBuffer, offset: 0, index: Int(NODE_CENTER_OF_MASS_IDX))
        encoder.setBuffer(self.nodeData.totalMassBuffer, offset: 0, index: Int(NODE_TOTAL_MASS_IDX))
        encoder.setBuffer(self.nodeData.startBuffer, offset: 0, index: Int(NODE_START_IDX))
        encoder.setBuffer(self.nodeData.endBuffer, offset: 0, index: Int(NODE_END_IDX))
        encoder.setBuffer(self.nodeData.isLeafBuffer, offset: 0, index: Int(NODE_IS_LEAF_IDX))
        encoder.setBuffer(self.mutexBuffer, offset: 0, index: Int(MUTEX_IDX))
        
        let gridSize = MTLSize(width: (Int(params.numNodes) + threadgroupSize.width - 1) / threadgroupSize.width,
                               height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
    
    private func computeBoundingBox(commandBuffer: borrowing MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        encoder.setComputePipelineState(boundingBoxPipeline)
        encoder.setBuffer(self.nodeData.topLeftBuffer, offset: 0, index: Int(NODE_TOP_LEFT_IDX))
        encoder.setBuffer(self.nodeData.bottomRightBuffer, offset: 0, index: Int(NODE_BOTTOM_RIGHT_IDX))
        encoder.setBuffer(self.bodyData.positionBuffer, offset: 0, index: Int(BODY_POSITION_IDX))
        encoder.setBuffer(mutexBuffer, offset: 0, index: Int(MUTEX_IDX))
        
        let gridSize = MTLSize(width: (Int(params.numBodies) + threadgroupSize.width - 1) / threadgroupSize.width,
                               height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
    
    private func constructQuadTree(commandBuffer: borrowing MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        encoder.setComputePipelineState(constructTreePipeline)
        
        var currentMass = self.bodyData.massBuffer
        var currentPosition = self.bodyData.positionBuffer
        var currentInitialIdx = self.bodyData.initialIdxBuffer
        
        var nextMass = self.bodyData.massAltBuffer
        var nextPosition = self.bodyData.positionAltBuffer
        var nextInitialIdx = self.bodyData.initialIdxAltBuffer
        
        encoder.setBuffer(self.nodeData.topLeftBuffer, offset: 0, index: Int(NODE_TOP_LEFT_IDX))
        encoder.setBuffer(self.nodeData.bottomRightBuffer, offset: 0, index: Int(NODE_BOTTOM_RIGHT_IDX))
        encoder.setBuffer(self.nodeData.centerOfMassBuffer, offset: 0, index: Int(NODE_CENTER_OF_MASS_IDX))
        encoder.setBuffer(self.nodeData.totalMassBuffer, offset: 0, index: Int(NODE_TOTAL_MASS_IDX))
        encoder.setBuffer(self.nodeData.startBuffer, offset: 0, index: Int(NODE_START_IDX))
        encoder.setBuffer(self.nodeData.endBuffer, offset: 0, index: Int(NODE_END_IDX))
        encoder.setBuffer(self.nodeData.isLeafBuffer, offset: 0, index: Int(NODE_IS_LEAF_IDX))
        
        for level in 0...self.params.maxDepth {
            let nodesInLevel = Int(pow(4.0, Double(level)))
            var nodeOffset = (Int32(pow(4.0, Double(level))) - 1) / 3
            
            encoder.setBuffer(currentMass, offset: 0, index: Int(BODY_MASS_IDX))
            encoder.setBuffer(currentPosition, offset: 0, index: Int(BODY_POSITION_IDX))
            encoder.setBuffer(currentInitialIdx, offset: 0, index: Int(BODY_INITIAL_IDX_IDX))

            encoder.setBuffer(nextMass, offset: 0, index: Int(BODY_MASS_ALT_IDX))
            encoder.setBuffer(nextPosition, offset: 0, index: Int(BODY_POSITION_ALT_IDX))
            encoder.setBuffer(nextInitialIdx, offset: 0, index: Int(BODY_INITIAL_IDX_ALT_IDX))

            encoder.setBytes(&nodeOffset, length: MemoryLayout<Int32>.stride, index: 22)
            
            let countMemSize = MemoryLayout<Int32>.stride * 8
            encoder.setThreadgroupMemoryLength(countMemSize, index: 0)
            
            let gridSize = MTLSize(width: nodesInLevel, height: 1, depth: 1)
            encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
            
            swap(&currentMass, &nextMass)
            swap(&currentPosition, &nextPosition)
            swap(&currentInitialIdx, &nextInitialIdx)
        }
        
        encoder.endEncoding()
        
        if self.params.maxDepth % 2 == 1 {
            swap(&currentMass, &nextMass)
            swap(&currentPosition, &nextPosition)
            swap(&currentInitialIdx, &nextInitialIdx)
        }
    }
    
    private func computeForces(commandBuffer: borrowing MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        encoder.setComputePipelineState(computeForcePipeline)
        
        encoder.setBuffer(self.nodeData.topLeftBuffer, offset: 0, index: Int(NODE_TOP_LEFT_IDX))
        encoder.setBuffer(self.nodeData.bottomRightBuffer, offset: 0, index: Int(NODE_BOTTOM_RIGHT_IDX))
        encoder.setBuffer(self.nodeData.centerOfMassBuffer, offset: 0, index: Int(NODE_CENTER_OF_MASS_IDX))
        encoder.setBuffer(self.nodeData.isLeafBuffer, offset: 0, index: Int(NODE_IS_LEAF_IDX))
        encoder.setBuffer(self.edgesData.edgeIndiciesBuffer, offset: 0, index: Int(EDGE_INDICIES_IDX))
        encoder.setBuffer(self.bodyData.massBuffer, offset: 0, index: Int(BODY_MASS_IDX))
        encoder.setBuffer(self.bodyData.radiusBuffer, offset: 0, index: Int(BODY_RADIUS_IDX))
        encoder.setBuffer(self.bodyData.positionBuffer, offset: 0, index: Int(BODY_POSITION_IDX))
        encoder.setBuffer(self.bodyData.velocityBuffer, offset: 0, index: Int(BODY_VELOCITY_IDX))
        encoder.setBuffer(self.bodyData.accelerationBuffer, offset: 0, index: Int(BODY_ACCELERATION_IDX))
        encoder.setBuffer(self.bodyData.initialIdxBuffer, offset: 0, index: Int(BODY_INITIAL_IDX_IDX))
        encoder.setBuffer(self.bodyData.offsetsBuffer, offset: 0, index: Int(BODY_OFFSETS_IDX))  // NEW: Add offsets buffer
        encoder.setBytes(&self.params.physics, length: MemoryLayout<PhysicsParams>.stride, index: Int(PHYSICS_PARAMS_IDX))
        
        let gridSize = MTLSize(width: (Int(params.numBodies) + threadgroupSize.width - 1) / threadgroupSize.width,
                               height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
    
    // MARK: - Draw
    public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) { }

    func runStage(name: String,
                  on commandBuffer : MTLCommandBuffer,
                  _ encode: (MTLCommandBuffer) -> Void) {
        encode(commandBuffer)
        if self.benchmarkOnly {
            commandBuffer.addCompletedHandler { completedBuffer in
                let startTime = completedBuffer.gpuStartTime
                let endTime = completedBuffer.gpuEndTime
                let gpuExecutionTime = (endTime - startTime) * 1000.0
                let namePadded = name.padding(toLength: 25, withPad: " ", startingAt: 0)
                print("\(namePadded) dispatch time: \(String(format: "%6.3f", gpuExecutionTime)) ms")

            }
        }
    }

    public func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable, let descriptor = view.currentRenderPassDescriptor else {
            return
        }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        
        defer { self.bodiesInitialized = true }
        if !self.bodiesInitialized {
            self.runStage(name: "Initialize Bodies", on: commandBuffer) { self.initalizeBodies(commandBuffer: $0) }
            self.runStage(name: "Initialize Edges", on: commandBuffer) { self.initalizeEdges(commandBuffer: $0) }

        }
        
        if self.params.useBarnes {
            self.runStage(name: "Reset Tree", on: commandBuffer) { self.resetTree(commandBuffer: $0) }
            self.runStage(name: "Compute Bounding Box", on: commandBuffer) { self.computeBoundingBox(commandBuffer: $0) }
            self.runStage(name: "Construct QuadTree", on: commandBuffer) { self.constructQuadTree(commandBuffer: $0) }
        }
    
        self.runStage(name: "Compute Forces", on: commandBuffer) { self.computeForces(commandBuffer: $0) }
        
        guard !self.benchmarkOnly else {
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            return
        }

        memcpy(self.transformBuffer.contents(), &self.screenTransform, MemoryLayout<ScreenTransform>.stride)
        
        let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor)!
        
        renderEncoder.setVertexBuffer(self.transformBuffer, offset: 0, index: Int(SCREEN_TRANSFORM_IDX))
        
        renderEncoder.setRenderPipelineState(self.nodeRenderPipeline)
        renderEncoder.setVertexBuffer(self.nodeData.topLeftBuffer, offset: 0, index: Int(NODE_TOP_LEFT_IDX))
        renderEncoder.setVertexBuffer(self.nodeData.bottomRightBuffer, offset: 0, index: Int(NODE_BOTTOM_RIGHT_IDX))
        
        renderEncoder.drawPrimitives(type: .line, vertexStart: 0, vertexCount: Int(self.params.numNodes) * 8)
        
        renderEncoder.setRenderPipelineState(self.bodyLineRenderPipeline)
        renderEncoder.setObjectBuffer(self.bodyData.positionBuffer, offset: 0, index: Int(BODY_POSITION_IDX))
        renderEncoder.setObjectBuffer(self.bodyData.offsetsBuffer, offset: 0, index: Int(BODY_OFFSETS_IDX))
        renderEncoder.setObjectBuffer(self.transformBuffer, offset: 0, index: Int(SCREEN_TRANSFORM_IDX))
        renderEncoder.setObjectBuffer(self.edgesData.edgeIndiciesBuffer, offset: 0, index: Int(EDGE_INDICIES_IDX))
        
        let gridSize = MTLSize(width: Int(params.numBodies), height: 1, depth: 1)
        let oneThread = MTLSize(width: 1, height: 1, depth: 1)
        renderEncoder.drawMeshThreadgroups(gridSize, threadsPerObjectThreadgroup: oneThread, threadsPerMeshThreadgroup: oneThread)
        
        renderEncoder.setRenderPipelineState(self.bodyRenderPipeline)
        renderEncoder.drawMeshThreadgroups(gridSize, threadsPerObjectThreadgroup: oneThread, threadsPerMeshThreadgroup: oneThread)
        
        renderEncoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
        
        commandBuffer.waitUntilCompleted()
        
        //self.printBodies()
    }
    
    // MARK: - Debug
    private func printTree() {
        let topLeftPtr = self.nodeData.topLeftBuffer.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        let bottomRightPtr = self.nodeData.bottomRightBuffer.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        let centerOfMassPtr = self.nodeData.centerOfMassBuffer.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        let totalMassPtr = self.nodeData.totalMassBuffer.contents().assumingMemoryBound(to: Float.self)
        let startPtr = self.nodeData.startBuffer.contents().assumingMemoryBound(to: UInt32.self)
        let endPtr = self.nodeData.endBuffer.contents().assumingMemoryBound(to: UInt32.self)
        let isLeafPtr = self.nodeData.isLeafBuffer.contents().assumingMemoryBound(to: Bool.self)
        
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
    
    /*
    private func printPerBodyEdgeIdx() {
        let edgesDataPointer = edgesDataBuffer.contents().assumingMemoryBound(to: EdgesData.self)
        
        print("\n=== EDGES DEBUG ===")
        withUnsafeMutablePointer(to: &edgesDataPointer.pointee.edges.0) { edgesPtr in
            for i in 0..<Int(edgesDataPointer.pointee.numBodiesWithEdges) {
                print(edgesPtr.advanced(by: i).pointee)
            }
        }
    }
    */
    
    private func printBodies() {
        //let massPtr = self.bodyData.massBuffer.contents().assumingMemoryBound(to: Float.self)
        //let radiusPtr = self.bodyData.radiusBuffer.contents().assumingMemoryBound(to: Float.self)
        let positionPtr = self.bodyData.positionBuffer.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        let velocityPtr = self.bodyData.velocityBuffer.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        let accelerationPtr = self.bodyData.accelerationBuffer.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        let initialIdxPtr = self.bodyData.initialIdxBuffer.contents().assumingMemoryBound(to: UInt32.self)
        let offsetBufferPtr = self.bodyData.offsetsBuffer.contents().assumingMemoryBound(to: UInt32.self)
        
        print("\n=== BODIES DEBUG ===")
        
        for i in 0..<Int(self.params.numBodies) {
            let position = positionPtr[i]
            let velocity = velocityPtr[i]
            let acceleration = accelerationPtr[i]
            let initialIdx = initialIdxPtr[i]
            let offsetIdx = offsetBufferPtr[i]
            
            print("\(i)]: Position=(\(position.x), \(position.y)), Velocity=(\(velocity.x), \(velocity.y)), Acceleration=(\(acceleration.x), \(acceleration.y)), InitialIdx = \(initialIdx), OffsetIdx = \(offsetIdx)")
        }
    }
    
    private func printNodes() {
        let centerOfMassPtr = self.nodeData.centerOfMassBuffer.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        
        print("\n=== NODES DEBUG ===")
        for i in 0..<min(10, Int(self.params.numNodes)) {
            let com = centerOfMassPtr[i]
            print("Node[\(i)]: CenterOfMass=(\(com.x), \(com.y))")
        }
    }
}
