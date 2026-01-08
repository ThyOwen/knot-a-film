import MetalKit
import SharedWithMetal

struct GraphPrintParams {
    let debugNodes : Bool = false
    let debugBodies : Bool = false
    let debugEdges : Bool = false
    let benchmark : Bool = false
    
    var isDebugEnabled : Bool {
        return self.debugNodes || self.debugBodies || self.debugEdges
    }
}
@Observable
public final class GraphRenderer : NSObject, MTKViewDelegate {
    
    @ObservationIgnored let device : MTLDevice
    @ObservationIgnored let commandQueue : MTLCommandQueue
    
    @ObservationIgnored private var initalizeEdgesPipeline : MTLComputePipelineState
    @ObservationIgnored private var initalizeBodiesPipeline : MTLComputePipelineState
    @ObservationIgnored private var resetPipeline : MTLComputePipelineState
    @ObservationIgnored private var boundingBoxPipeline : MTLComputePipelineState
    @ObservationIgnored private var constructTreePipeline : MTLComputePipelineState
    @ObservationIgnored private var computeForcePipeline : MTLComputePipelineState
    @ObservationIgnored private var sortEdgeAnglesPipeline : MTLComputePipelineState
    
    @ObservationIgnored private var nodeRenderPipeline : MTLRenderPipelineState
    @ObservationIgnored private var bodyRenderPipeline : MTLRenderPipelineState
    @ObservationIgnored private var bodyLineRenderPipeline : MTLRenderPipelineState
    
    @ObservationIgnored private var nodeData : NodeBufferGroup
    @ObservationIgnored private var bodyData : BodyBufferGroup
    @ObservationIgnored private var edgesData : EdgesBufferGroup
    
    @ObservationIgnored private var mutexBuffer : MTLBuffer
    @ObservationIgnored private var transformBuffer : MTLBuffer
    
    @ObservationIgnored private var edgesBuffer : MTLBuffer
    @ObservationIgnored private var offsetsBuffer : MTLBuffer
            
    @ObservationIgnored private var params : GraphParams
    @ObservationIgnored private let printParams : GraphPrintParams
    
    @ObservationIgnored var bodiesInitialized : Bool = false

    var prevIndicies : [UInt32] = []
    var bodyPositions : [(index: Int, position: CGPoint)] = []
    
    public var screenTransform : ScreenTransform = .init(offset: .zero, scale: .init(1.0, 1.0))
    
    private let threadgroupSize = MTLSize(width: 32, height: 1, depth: 1)
    
    public init(edges: [UInt32], offsets: consuming [UInt32]) {
        
        if edges.isEmpty || offsets.isEmpty {
            fatalError("inputs are empty")
        }
        
        let numBodies = UInt32(offsets.count)
        let numEdges = UInt32(edges.count)

        
        self.params = GraphParams.makeDefault(numBodies: numBodies, numEdges: numEdges)
        self.printParams = GraphPrintParams()
        
        let device = MTLCreateSystemDefaultDevice()!
        
        let functionConstants = MTLFunctionConstantValues()
        
        self.params.addMetalFunctionConstants(to: functionConstants)
        
        //create bodies
        self.bodyData = BodyBufferGroup.makeBuffers(size: numBodies, device: device, options: .storageModeShared)
        self.bodyData.setSentinal(numBodies, numEdges)
        
        //create nodes
        let numNodes = UInt32(self.params.numNodes)
        self.nodeData = NodeBufferGroup.makeBuffers(size: numNodes, device: device, options: .storageModeShared)
        
        self.edgesData = EdgesBufferGroup.makeBuffers(size: numEdges, device: device, options: .storageModeShared)

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
        //let library = try! device.makeDefaultLibrary(bundle: .main)
        
        let nodePipelineDescriptor = MTLRenderPipelineDescriptor()
        nodePipelineDescriptor.vertexFunction = try! library.makeFunction(name: "vertexNode", constantValues: functionConstants)
        nodePipelineDescriptor.fragmentFunction = try! library.makeFunction(name: "fragmentNode", constantValues: functionConstants)
        nodePipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        
        let bodyPipelineDescriptor = MTLMeshRenderPipelineDescriptor()
        bodyPipelineDescriptor.meshFunction = try! library.makeFunction(name: "meshPointShader", constantValues: functionConstants)
        bodyPipelineDescriptor.objectFunction = try! library.makeFunction(name: "objectShader", constantValues: functionConstants)
        bodyPipelineDescriptor.fragmentFunction = try! library.makeFunction(name: "fragmentBody", constantValues: functionConstants)
        bodyPipelineDescriptor.maxTotalThreadsPerObjectThreadgroup = 32
        bodyPipelineDescriptor.maxTotalThreadsPerMeshThreadgroup = 32
        bodyPipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        
        self.nodeRenderPipeline = try! device.makeRenderPipelineState(descriptor: nodePipelineDescriptor)
        
        let pipelineOption : MTLPipelineOption = .failOnBinaryArchiveMiss
        let (bodyRenderPipeline, _) = try! device.makeRenderPipelineState(descriptor: bodyPipelineDescriptor, options: pipelineOption)
        self.bodyRenderPipeline = bodyRenderPipeline
        
        let bodyLinePipelineDescriptor = MTLMeshRenderPipelineDescriptor()
        bodyLinePipelineDescriptor.meshFunction = try! library.makeFunction(name: "meshLineShader", constantValues: functionConstants)
        bodyLinePipelineDescriptor.objectFunction = try! library.makeFunction(name: "objectShader", constantValues: functionConstants)
        bodyLinePipelineDescriptor.fragmentFunction = try! library.makeFunction(name: "fragmentBody", constantValues: functionConstants)
        bodyLinePipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        bodyLinePipelineDescriptor.maxTotalThreadsPerObjectThreadgroup = 32
        bodyLinePipelineDescriptor.maxTotalThreadsPerMeshThreadgroup = 32
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
        
        let sortEdgeAnglesFunction = try! library.makeFunction(name: "sortEdgeAngles", constantValues: functionConstants)
        self.sortEdgeAnglesPipeline = try! device.makeComputePipelineState(function: sortEdgeAnglesFunction)
        
        self.commandQueue = device.makeCommandQueue()!

        self.device = device
        super.init()
    }
    
    // MARK: - Kernels
    private func initalizeEdges(commandBuffer: borrowing MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        encoder.setComputePipelineState(initalizeEdgesPipeline)
        
        encoder.setBuffer(self.edgesBuffer, offset: 0, index: 0)
        encoder.setBuffer(self.offsetsBuffer, offset: 0, index: 1)
        encoder.setBuffer(self.bodyData.offsetsBuffer, offset: 0, index: Int(BODY_EDGE_OFFSETS_IDX))
        encoder.setBuffer(self.edgesData.edgeTerminationsBuffer, offset: 0, index: Int(EDGE_TERMINATIONS_IDX))
        encoder.setBuffer(self.edgesData.edgeSourcesBuffer, offset: 0, index: Int(EDGE_SOURCES_IDX))

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
        
        let gridSize = MTLSize(width: Int(params.numBodies), height: 1, depth: 1)

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
        
        let gridSize = MTLSize(width: Int(params.numNodes), height: 1, depth: 1)

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
        encoder.setBuffer(self.edgesData.edgeTerminationsBuffer, offset: 0, index: Int(EDGE_TERMINATIONS_IDX))
        encoder.setBuffer(self.bodyData.massBuffer, offset: 0, index: Int(BODY_MASS_IDX))
        encoder.setBuffer(self.bodyData.radiusBuffer, offset: 0, index: Int(BODY_RADIUS_IDX))
        encoder.setBuffer(self.bodyData.positionBuffer, offset: 0, index: Int(BODY_POSITION_IDX))
        encoder.setBuffer(self.bodyData.velocityBuffer, offset: 0, index: Int(BODY_VELOCITY_IDX))
        encoder.setBuffer(self.bodyData.accelerationBuffer, offset: 0, index: Int(BODY_ACCELERATION_IDX))
        encoder.setBuffer(self.bodyData.initialIdxBuffer, offset: 0, index: Int(BODY_INITIAL_IDX_IDX))
        encoder.setBuffer(self.bodyData.offsetsBuffer, offset: 0, index: Int(BODY_EDGE_OFFSETS_IDX))
        encoder.setBuffer(self.edgesData.edgeAnglesBuffer, offset: 0, index: Int(EDGE_ANGLES_IDX))
        encoder.setBytes(&self.params.physics, length: MemoryLayout<PhysicsParams>.stride, index: Int(PHYSICS_PARAMS_IDX))
        
        let perBodyAnglesMemSize = MemoryLayout<Float32>.stride * Int(MAX_EDGES)
        encoder.setThreadgroupMemoryLength(perBodyAnglesMemSize, index: 0)
        
        let gridSize = MTLSize(width: Int(params.numBodies), height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
    
    private func sortEdgeAngles(commandBuffer: borrowing MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        encoder.setComputePipelineState(self.sortEdgeAnglesPipeline)
        
        encoder.setBuffer(self.bodyData.offsetsBuffer, offset: 0, index: Int(BODY_EDGE_OFFSETS_IDX))
        encoder.setBuffer(self.edgesData.edgeAnglesBuffer, offset: 0, index: Int(EDGE_ANGLES_IDX))
        encoder.setBuffer(self.edgesData.edgeTerminationsBuffer, offset: 0, index: Int(EDGE_TERMINATIONS_IDX))
        encoder.setBuffer(self.edgesData.edgeTerminationsSortedBuffer, offset: 0, index: Int(EDGE_TERMINATIONS_SORTED_IDX))

        let gridSize = MTLSize(width: Int(params.numBodies), height: 1, depth: 1)

        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
    
    private func render(with descriptor : borrowing MTLRenderPassDescriptor, commandBuffer: borrowing MTLCommandBuffer) {
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else {
            return
        }
        
        memcpy(self.transformBuffer.contents(), &self.screenTransform, MemoryLayout<ScreenTransform>.stride)
                
        renderEncoder.setVertexBuffer(self.transformBuffer, offset: 0, index: Int(SCREEN_TRANSFORM_IDX))
        
        renderEncoder.setRenderPipelineState(self.nodeRenderPipeline)
        renderEncoder.setVertexBuffer(self.nodeData.topLeftBuffer, offset: 0, index: Int(NODE_TOP_LEFT_IDX))
        renderEncoder.setVertexBuffer(self.nodeData.bottomRightBuffer, offset: 0, index: Int(NODE_BOTTOM_RIGHT_IDX))
        
        renderEncoder.drawPrimitives(type: .line, vertexStart: 0, vertexCount: Int(self.params.numNodes) * 8)
        print("draw")

        renderEncoder.setRenderPipelineState(self.bodyLineRenderPipeline)
        renderEncoder.setObjectBuffer(self.edgesData.edgeTerminationsSortedBuffer, offset: 0, index: Int(EDGE_TERMINATIONS_SORTED_IDX))
        renderEncoder.setObjectBuffer(self.edgesData.edgeSourcesBuffer, offset: 0, index: Int(EDGE_SOURCES_IDX))
        renderEncoder.setObjectBuffer(self.edgesData.edgeTerminationsBuffer, offset: 0, index: Int(EDGE_TERMINATIONS_IDX))

        renderEncoder.setObjectBuffer(self.bodyData.positionBuffer, offset: 0, index: Int(BODY_POSITION_IDX))
        renderEncoder.setObjectBuffer(self.bodyData.offsetsBuffer, offset: 0, index: Int(BODY_EDGE_OFFSETS_IDX))
        renderEncoder.setObjectBuffer(self.transformBuffer, offset: 0, index: Int(SCREEN_TRANSFORM_IDX))
        
        let gridSize = MTLSize(width: (Int(params.numEdges) + threadgroupSize.width - 1) / threadgroupSize.width, height: 1, depth: 1)
        renderEncoder.drawMeshThreadgroups(gridSize, threadsPerObjectThreadgroup: threadgroupSize, threadsPerMeshThreadgroup: threadgroupSize)
        
        //renderEncoder.setRenderPipelineState(self.bodyRenderPipeline)
        //renderEncoder.drawMeshThreadgroups(gridSize, threadsPerObjectThreadgroup: threadgroupSize, threadsPerMeshThreadgroup: threadgroupSize)
        
        renderEncoder.endEncoding()
    }
    
    // MARK: - Draw
    public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) { }

    public func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable, let descriptor = view.currentRenderPassDescriptor else {
            return
        }
        
        guard let commandBuffer = self.commandQueue.makeCommandBuffer() else { return }
        
        if !self.bodiesInitialized {
            self.runStage(name: "Initialize Bodies", on: commandBuffer) { self.initalizeBodies(commandBuffer: $0) }
            self.runStage(name: "Initialize Edges", on: commandBuffer) { self.initalizeEdges(commandBuffer: $0) }
            self.bodiesInitialized = true
        }
        
        if self.params.useBarnes {
            self.runStage(name: "Reset Tree", on: commandBuffer) { self.resetTree(commandBuffer: $0) }
            self.runStage(name: "Compute Bounding Box", on: commandBuffer) { self.computeBoundingBox(commandBuffer: $0) }
            self.runStage(name: "Construct QuadTree", on: commandBuffer) { self.constructQuadTree(commandBuffer: $0) }
        }
    
        self.runStage(name: "Compute Forces", on: commandBuffer) { self.computeForces(commandBuffer: $0) }
        self.runStage(name: "Sort Angles", on: commandBuffer) { self.sortEdgeAngles(commandBuffer: $0) }
        
        self.runStage(name: "Render", on: commandBuffer) { self.render(with: descriptor, commandBuffer: $0) }

        commandBuffer.present(drawable)
        commandBuffer.commit()
        
        self.updateBodyPositions()
        
        if self.printParams.isDebugEnabled {
            commandBuffer.waitUntilCompleted()
        }
                
        if self.printParams.debugNodes {
            self.printNodes()
        }
        if self.printParams.debugBodies {
            self.printBodies()
        }
        if self.printParams.debugEdges {
            self.printEdges()
        }
    }
    
    // MARK: - Debug
    
    public func updateBodyPositions() {
        let positionBase = self.bodyData.positionBuffer.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        let positionPtr = UnsafeBufferPointer(start: positionBase.advanced(by: 1), count: Int(params.numBodies))
        
        self.bodyPositions = (0..<Int(params.numBodies)).map { i in
            let pos = positionPtr[i]
            let screenX = CGFloat(pos.x * screenTransform.scale.x + screenTransform.offset.x)
            let screenY = CGFloat(pos.y * screenTransform.scale.y + screenTransform.offset.y)
            return (index: i, position: CGPoint(x: screenX, y: screenY))
        }
    }

    private func runStage(name: String, on commandBuffer : MTLCommandBuffer, _ encode: (MTLCommandBuffer) -> Void) {
        encode(commandBuffer)
        if self.printParams.benchmark {
            commandBuffer.addCompletedHandler { completedBuffer in
                let startTime = completedBuffer.gpuStartTime
                let endTime = completedBuffer.gpuEndTime
                let gpuExecutionTime = (endTime - startTime) * 1000.0
                let namePadded = name.padding(toLength: 25, withPad: " ", startingAt: 0)
                print("\(namePadded) dispatch time: \(String(format: "%6.3f", gpuExecutionTime)) ms")

            }
        }
    }

    private func printNodes() {
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
    
    private func printEdges() {
        let edgeIdxPtr = self.edgesData.edgeTerminationsBuffer.contents().assumingMemoryBound(to: UInt32.self)
        let sortedEdgeIdxPtr = self.edgesData.edgeTerminationsSortedBuffer.contents().assumingMemoryBound(to: UInt32.self)
        let edgeAnglesPtr = self.edgesData.edgeAnglesBuffer.contents().assumingMemoryBound(to: UInt32.self)
        let bodyIdxPtr = self.edgesData.edgeSourcesBuffer.contents().assumingMemoryBound(to: UInt32.self)
        let count = Int(self.params.numEdges)

        var indices : [UInt32] = []
        print("\n=== EDGES DEBUG ===")
        for i in 0...count {
            let edgeIdx = edgeIdxPtr[i]
            let sortedEdgeIdx = sortedEdgeIdxPtr[i]
            let angle = edgeAnglesPtr[i]
            let bodyIdx = bodyIdxPtr[i]
            
            indices.append(sortedEdgeIdx)

            print("""
            [\(i)] EdgeIdx=\(edgeIdx), \
            SortedEdgeIdx=\(sortedEdgeIdx), \
            Angle=\(angle), \
            BodyIdx=\(bodyIdx)
            """)
        }
        
        
        if !self.prevIndicies.isEmpty {
            zip(indices, self.prevIndicies).enumerated().forEach { idx, val in
                let current = val.0
                let previous = val.1
                if current != previous {
                    print("\n=== INDEX CHANGE ===")
                    print("\(idx): \(current) -> \(previous)")
                }
            }
        }
        
        self.prevIndicies = indices
    }

    private func printBodies() {
        //let massPtr = self.bodyData.massBuffer.contents().assumingMemoryBound(to: Float.self)
        //let radiusPtr = self.bodyData.radiusBuffer.contents().assumingMemoryBound(to: Float.self)

        let positionBase = self.bodyData.positionBuffer.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        let positionPtr = UnsafeBufferPointer(start: positionBase.advanced(by: 1), count: Int(params.numBodies))
        
        let velocityBase = self.bodyData.velocityBuffer.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        let velocityPtr = UnsafeBufferPointer(start: velocityBase.advanced(by: 1), count: Int(params.numBodies))
        
        let accelerationBase = self.bodyData.accelerationBuffer.contents().assumingMemoryBound(to: SIMD2<Float>.self)
        let accelerationPtr = UnsafeBufferPointer(start: accelerationBase.advanced(by: 1), count: Int(params.numBodies))
        
        let initialIdxBase = self.bodyData.initialIdxBuffer.contents().assumingMemoryBound(to: UInt32.self)
        let initialIdxPtr = UnsafeBufferPointer(start: initialIdxBase.advanced(by: 1), count: Int(params.numBodies))
        
        let offsetBufferBase = self.bodyData.offsetsBuffer.contents().assumingMemoryBound(to: UInt32.self)
        let offsetBufferPtr = UnsafeBufferPointer(start: offsetBufferBase.advanced(by: 1), count: Int(params.numBodies) + 1)
        
        print("\n=== BODIES DEBUG ===")
        for i in 0..<Int(self.params.numBodies) {
            let position = positionPtr[i]
            let velocity = velocityPtr[i]
            let acceleration = accelerationPtr[i]
            let initialIdx = initialIdxPtr[i]
            let offsetIdx = offsetBufferPtr[i]
            
            print("\(i)]: Position=(\(position.x), \(position.y)), Velocity=(\(velocity.x), \(velocity.y)), Acceleration=(\(acceleration.x), \(acceleration.y)), InitialIdx = \(initialIdx), OffsetIdx = \(offsetIdx)")
        }
        
        // Print sentinel
        print("Sentinel: offsetBufferPtr[\(params.numBodies)] = \(offsetBufferPtr[Int(params.numBodies)])")
    }
}
