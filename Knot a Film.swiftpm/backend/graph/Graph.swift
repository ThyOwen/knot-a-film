//
//  Graph.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 9/7/25.
//

import MetalPerformanceShadersGraph


struct Graph {
    
    let graphParams : GraphParams
    
    let graph : MPSGraph = .init()
    var graphExe : MPSGraphExecutable?
    
    let numNodes : NSNumber
    let numConnections : NSNumber
    
    private var outputPositions : MPSGraphTensor?
    private var outputVelocities : MPSGraphTensor?
    private var tensorOfInterest : MPSGraphTensor?
        
    private var positionsTensor : MPSGraphTensor
    private var velocitiesTensor : MPSGraphTensor
    private var connectionsTensor : MPSGraphTensor
    
    var scratchBuffer : ContiguousArray<Float32> = []
    let connections : [SIMD2<UInt32>]

    init(graphParams : GraphParams, connections : [SIMD2<UInt32>]) {
        
        let numNodes = NSNumber(value: graphParams.numNodes)
        let numConnections = NSNumber(integerLiteral: connections.count)

        self.positionsTensor = self.graph.placeholder(shape: [numNodes, 2], dataType: .float32, name: "positions") // [N, 2]
        self.velocitiesTensor = self.graph.placeholder(shape: [numNodes, 2], dataType: .float32, name: "velocities") // [N, 2]
        self.connectionsTensor = self.graph.placeholder(shape: [numConnections, 2], dataType: .int32, name: "connections") // [C, 2]
        
        self.numConnections = numConnections
        self.numNodes = numNodes
        self.graphParams = graphParams
        self.connections = connections

    }
    
    mutating func testRun(positionsBuffer : inout MTLBuffer,
                          velocitiesBuffer : inout MTLBuffer,
                          connectionsBuffer : inout MTLBuffer) {
                
        guard let outputVelocities = self.outputVelocities,
              let outputPositions = self.outputPositions,
              let tensorOfInterest = self.tensorOfInterest else {
            fatalError("tenors not initalized")
        }
        
        let positionsTensorData = MPSGraphTensorData(positionsBuffer, shape: [numNodes, 2], dataType: .float32)
        let velocitiesTensorData = MPSGraphTensorData(velocitiesBuffer, shape: [numNodes, 2], dataType: .float32)
        let connectionsTensorData = MPSGraphTensorData(connectionsBuffer, shape: [numConnections, 2], dataType: .int32)
        
        let inputFeeds : [MPSGraphTensor: MPSGraphTensorData] = [
            positionsTensor : positionsTensorData,
            velocitiesTensor : velocitiesTensorData,
            connectionsTensor : connectionsTensorData
        ]
        
        
        let results = self.graph.run(
            feeds: inputFeeds,
            targetTensors: [outputVelocities, outputPositions, tensorOfInterest],
            targetOperations: nil
        )
        
        // Extract result
        
        guard let resultPositions : MPSGraphTensorData = results[outputPositions] else {
            fatalError("No result")
        }
        
        guard let resultVelocities : MPSGraphTensorData = results[outputVelocities] else {
            fatalError("No result")
        }
        
        resultPositions.mpsndarray().readBytes(positionsBuffer.contents(), strideBytes: nil)
        resultVelocities.mpsndarray().readBytes(velocitiesBuffer.contents(), strideBytes: nil)
        
        if false {
            guard let resultTensor : MPSGraphTensorData = results[tensorOfInterest] else {
                fatalError("No result")
            }
            
            self.printBuffer(outputTensor: tensorOfInterest, outputTensorData: resultTensor)
        }
    }
    
    mutating func runInplace(positionsBuffer : inout MTLBuffer, velocitiesBuffer : inout MTLBuffer, connectionsBuffer : inout MTLBuffer, on commandQueue : MTLCommandQueue, printBuffers : Bool = false) {
                
        guard let outputVelocities = self.outputVelocities,
                let outputPositions = self.outputPositions else {
            fatalError("tenors not initalized")
        }


        let positionsTensorData = MPSGraphTensorData(positionsBuffer, shape: [numNodes, 2], dataType: .float32)
        let velocitiesTensorData = MPSGraphTensorData(velocitiesBuffer, shape: [numNodes, 2], dataType: .float32)
        let connectionsTensorData = MPSGraphTensorData(connectionsBuffer, shape: [numConnections, 2], dataType: .int32)
        
        let updatedPositionsTensorData = MPSGraphTensorData(positionsBuffer, shape: [numNodes, 2], dataType: .float32)
        let updatedVelocitiesTensorData = MPSGraphTensorData(velocitiesBuffer, shape: [numNodes, 2], dataType: .float32)
        
        let commandBuffer = MPSCommandBuffer(from: commandQueue)
        
        let inputFeeds : [MPSGraphTensor: MPSGraphTensorData] = [
            self.positionsTensor : positionsTensorData,
            self.velocitiesTensor : velocitiesTensorData,
            self.connectionsTensor : connectionsTensorData,
        ]
        
        let outputFeeds : [MPSGraphTensor: MPSGraphTensorData] = [
            outputPositions : updatedPositionsTensorData,
            outputVelocities : updatedVelocitiesTensorData,
        ]

        let executionOptions = MPSGraphExecutionDescriptor()
    
        self.graph.encode(
            to: commandBuffer,
            feeds: inputFeeds,
            targetOperations: nil,
            resultsDictionary: outputFeeds,
            executionDescriptor: executionOptions
        )

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if printBuffers {
            print("positions")
            self.printBuffer(outputTensor: outputPositions, outputTensorData: updatedPositionsTensorData)
            print("velocities")
            self.printBuffer(outputTensor: outputVelocities, outputTensorData: updatedVelocitiesTensorData)

        }
      
    }
    
    mutating func compile(on metalDevice : MTLDevice) {
        
        guard let outputVelocities = self.outputVelocities,
              let outputPositions = self.outputPositions else {
            fatalError("tenors not initalized")
        }
        
        let inputFeeds : [MPSGraphTensor : MPSGraphShapedType] = [
            positionsTensor : .init(shape: [numNodes, 2], dataType: .float32),
            velocitiesTensor : .init(shape: [numNodes, 2], dataType: .float32),
            connectionsTensor : .init(shape: [numConnections, 2], dataType: .int32)
        ]
        
        let graphDevice = MPSGraphDevice(mtlDevice: metalDevice)
        
        let compilationDescriptor = MPSGraphCompilationDescriptor()
        //compilationDescriptor.reducedPrecisionFastMath = .allowFP16Intermediates
        
        compilationDescriptor.optimizationLevel = .level1
        compilationDescriptor.waitForCompilationCompletion = true
        compilationDescriptor.reducedPrecisionFastMath = .allowFP16Intermediates
        compilationDescriptor.reducedPrecisionFastMath = .allowFP16Conv2DWinogradTransformIntermediate
        
        let exe = self.graph.compile(with: graphDevice,
                           feeds: inputFeeds,
                           targetTensors: [outputPositions, outputVelocities],
                           targetOperations: [],
                           compilationDescriptor: compilationDescriptor)
        
        exe.serialize(package: .downloadsDirectory.appending(component: "test.mpsgraphpackage"), descriptor: nil)
        
        self.graphExe = exe
    }
    
    
    // builders
    mutating func buildGraphStandard(sparse : Bool = false) {


        let positionsTensor = self.graph.placeholder(shape: [self.numNodes, 2], dataType: .float32, name: "positions") // [N, 2]
        let velocityTensor = self.graph.placeholder(shape: [self.numNodes, 2], dataType: .float32, name: "velocities") // [N, 2]
        let connectionsTensor = self.graph.placeholder(shape: [self.numConnections, 2], dataType: .int32, name: "connections")
        
        self.positionsTensor = positionsTensor
        self.velocitiesTensor = velocityTensor
        
        //the actual function
        
        let aNodePositions = self.graph.expandDims(positionsTensor, axes: [1], name: "a_node_positions_expanded") // [N, 1, 2]
        let bNodePositions = self.graph.expandDims(positionsTensor, axes: [0], name: "b_node_positions_expanded") // [1, N, 2]
        
        let aNodePositionsTiled = self.graph.tileTensor(aNodePositions, withMultiplier: [1, numNodes, 1], name: "a_node_positions_tiled") // [N, N, 2]
        let bNodePositionsTiled = self.graph.tileTensor(bNodePositions, withMultiplier: [numNodes, 1, 1], name: "b_node_positions_tiled") // [N, N, 2]
        
        //make sparse tensors
        
        let triangularIndices = Self.getTriangularIndices(length: numNodes.int32Value)
        
        let deltas : MPSGraphTensor
        
        if sparse {
            let (aNodePositionsSparse, bNodePositionsSparse) = self.withTupledAccess(to: triangularIndices) { columnTensor, rowTensor in
                let indices = self.graph.stack([rowTensor, columnTensor], axis: 1, name: nil)
                
                let buildSparseTensor : (MPSGraphTensor) -> MPSGraphTensor = { [self] nodePositions in
                    let nodePositionsSparseAxisSplit : [MPSGraphTensor] = (0..<2).map { idx in
                        
                        let start = NSNumber(value: idx)
                        let end = NSNumber(value: idx + 1)
                        
                        let nodeAxisSlice = self.graph.sliceTensor(nodePositions,
                                                                   starts: [0, 0, start],
                                                                   ends: [numNodes, numNodes, end],
                                                                   strides: [1,1,1], name: nil) // [N, N, 1]
                        
                        let nodeAxisSliceSqueezed = self.graph.squeeze(nodeAxisSlice, axis: 2, name: nil) // [N, N]
                        
                        let nodeAxisGathered = self.graph.gatherND(withUpdatesTensor: nodeAxisSliceSqueezed,
                                                                   indicesTensor: indices,
                                                                   batchDimensions: 0,
                                                                   name: nil) // [2, N_masked]
                        
                        let nodeAxisSliceSparse = self.graph.sparseTensor(sparseTensorWithType: .COO,
                                                                          tensors: [nodeAxisGathered, rowTensor, columnTensor],
                                                                          shape: [numNodes, numNodes],
                                                                          dataType: .float32,
                                                                          name: nil) // [N*, N*] * for sparse
                        return nodeAxisSliceSparse
                    }
                    
                    return self.graph.stack(nodePositionsSparseAxisSplit, axis: 2, name: nil)
                }
                
                let aNodePositionsSparse : MPSGraphTensor = buildSparseTensor(aNodePositionsTiled)
                let bNodePositionsSparse : MPSGraphTensor = buildSparseTensor(bNodePositionsTiled)
                
                return (aNodePositionsSparse, bNodePositionsSparse)
            }
            
            deltas = self.graph.subtraction(bNodePositionsSparse, aNodePositionsSparse, name: "positions_delta") // [N, N, 2]
        } else {
            deltas = self.graph.subtraction(bNodePositions, aNodePositions, name: "positions_delta") // [N, N, 2]
        }
        
        let deltaSquared = self.graph.square(with: deltas, name: "positions_delta_squared") // [N, N, 2]
        let distanceSquared = self.graph.reductionSum(with: deltaSquared, axis: 2, name: "distance_squared") // [N, N, 1]
        let epsilon = self.graph.constant(self.graphParams.epsilon, dataType: .float32) // [N]
        let softenedDistanceSquared = self.graph.addition(distanceSquared, epsilon, name: "softer_distance_squared") // [N, N, 1]
        let distances = self.graph.squareRoot(with: softenedDistanceSquared, name: "distances") // [N, N, 1]
                
        let directions = self.graph.divisionNoNaN(deltas, distances, name: "directions") // [N, N, 2]
        
        //calculate forces between every node and every other node
        let edgeRepulsionTensor = self.graph.constant(self.graphParams.edgeRepulsion, dataType: .float32)
        let repulsionForce = self.graph.divisionNoNaN(edgeRepulsionTensor, distances, name: "repulsion_force") // [N, N, 2]
                
        let edgeAttractionTensor = self.withTupledAccess(to: self.connections.map { (Int32($0.x), Int32($0.y)) }) { columnTensor, rowTensor in
            let edgeAttractionArray = Array(repeating: Float32(self.graphParams.edgeAttraction), count: numConnections.intValue)
            
            let edgeAttractionTensor = self.buildTensor(from: edgeAttractionArray, shape: [numConnections])
            
            let edgeAttractionSparse = self.graph.sparseTensor(sparseTensorWithType: .COO,
                                                               tensors: [edgeAttractionTensor, rowTensor, columnTensor],
                                                               shape: [numNodes, numNodes],
                                                               dataType: .float32,
                                                               name: nil) // [N, N]
            let edgeAttractionExpanded = self.graph.expandDims(edgeAttractionSparse, axis: 2, name: nil)
            
            return edgeAttractionExpanded
        } // [N, N, 1]

        let attractionForce = self.graph.divisionNoNaN(distances, edgeAttractionTensor, name: "attraction_force") // [N, N, 1]

        let forces = self.graph.subtraction(repulsionForce, attractionForce, name: "force") // [N, N, 1]
        
        let acceleration = self.graph.multiplication(forces, directions, name: "acceleration") // [N, N, 2] change in velocity F/m; m = 1 so just a
        
        let perNodeAcceleration : MPSGraphTensor
        
        if sparse {
            let aNodeAcceleration = self.graph.reductionSum(with: acceleration, axis: 0, name: "a_node_acceleration") // [N, 1, 2]
            
            let bNodeAcceleration = self.graph.reductionSum(with: acceleration, axis: 1, name: "b_node_acceleration") // [1, N, 2]
            let bNodeAccelerationTransposed = self.graph.transposeTensor(bNodeAcceleration, dimension: 1, withDimension: 0, name: "b_node_acceleration_transposed") // [1, N, 2]
            
            perNodeAcceleration = self.graph.subtraction(aNodeAcceleration, bNodeAccelerationTransposed, name: "per_node_acceleration") // [N, 1, 2]
        } else {
            perNodeAcceleration = self.graph.reductionSum(with: acceleration, axis: 0, name: "per_node_acceleration") // [N, 1, 2]
        }

        let nanMask = self.graph.isNaN(with: perNodeAcceleration, name: "nan_mask") // [bool]
        let zero = self.graph.constant(0.0, dataType: .float32) // [1]
        let safePerNodeAcceleration = self.graph.select(predicate: nanMask, trueTensor: zero, falseTensor: perNodeAcceleration, name: "safe_acceleration") // [N, 1, 2]

        let perNodeAccelerationSqueezed = self.graph.squeeze(safePerNodeAcceleration, axis: 0, name: "per_node_acceleration_squeezed") // [N, 2]
        
        let updatedVelocities = self.graph.addition(velocityTensor, perNodeAccelerationSqueezed, name: "updated_velocities") // [N, 2]
        
        let damping = self.graph.constant(self.graphParams.damping, dataType: .float32) // [1]
        let dampedVelocities = self.graph.multiplication(updatedVelocities, damping, name: "damped_velocities") // [N, 2]
        
        
        let updatedPositions = self.graph.addition(positionsTensor, dampedVelocities, name: "updated_positions") // [N, 2]

        self.outputVelocities = dampedVelocities
        self.outputPositions = updatedPositions
    }

    // utilities
    func withTupledAccess<T, N : Numeric>(to indices : consuming [(N, N)], completion : (_ : borrowing MPSGraphTensor, _ : borrowing MPSGraphTensor) -> T) -> T {
        let length = NSNumber(value: indices.count)
        
        let columnTensor = self.buildTensor(from: indices.map(\.0), shape: [length])
        let rowTensor = self.buildTensor(from: indices.map(\.1), shape: [length])
        
        return completion(columnTensor, rowTensor)
        
    }
    
    func withTupledAccess<T, N : Numeric>(to indices : consuming [SIMD2<N>], completion : (_ : borrowing MPSGraphTensor, _ : borrowing MPSGraphTensor) -> T) -> T {
        let length = NSNumber(value: indices.count)
        
        let columnTensor = self.buildTensor(from: indices.map(\.x), shape: [length])
        let rowTensor = self.buildTensor(from: indices.map(\.y), shape: [length])
        
        return completion(columnTensor, rowTensor)
        
    }
    
    func buildTensor<T : Numeric>(from data : borrowing [T], shape: borrowing [NSNumber]) -> MPSGraphTensor {
        let mpsType = T.mpsDataType()
        print("Using MPS type \(mpsType.description) from Swift type \(T.self)")
        
        //let outputLength = Self.getFlattenedSize(from: shape)
        
        //guard outputLength == data.count else {
            //fatalError("Shape \(shape) does not match data length \(data.count)")
        //}
        
        return data.withUnsafeBufferPointer { bufferPointer -> MPSGraphTensor in
            self.graph.constant(Data(buffer: bufferPointer), shape: shape, dataType: mpsType)
        }
        
    }

    static func getTriangularIndices(length : Int32) -> [(Int32, Int32)] {

        let size = Self.getNumRelevantNodes(numNodes: length).intValue
        
        var indices : [(Int32, Int32)] = .init()
        indices.reserveCapacity(size)
        
        for aIdx in 0..<length {
            for bIdx in (aIdx + 1)..<length {
                indices.append((Int32(aIdx), Int32(bIdx)))
            }
        }
        return indices
    }
    
    static func getRandomEdgeIndices(length : Int32) -> [(Int32, Int32)] {
        let numRandomGeneratedEdges = Int32.random(in: 0..<length)
        
        let randomEdgeIndices = (0..<numRandomGeneratedEdges).map { _ in
            let aIdx = Int32.random(in: 0..<length)
            let bIdx = Int32.random(in: 0..<length)
            return (aIdx, bIdx)
        }
        
        return randomEdgeIndices
        
    }
    
    mutating func printBuffer(outputTensor : MPSGraphTensor, outputTensorData : MPSGraphTensorData) {
        let outputShape = outputTensor.shape!
        
        let outputLength = Self.getFlattenedSize(from: outputShape)

        if self.scratchBuffer.count < outputLength {
            self.scratchBuffer = .init(repeating: 0.0, count: outputLength)
        }
        
        self.scratchBuffer.withUnsafeMutableBytes { pointer in
            guard let address = pointer.baseAddress else {
                fatalError("base address nil")
            }
            outputTensorData.mpsndarray().readBytes(address, strideBytes: nil)
        }
                
        let shape = outputShape.map(\.intValue)
        print("Shape: \(shape)", "Data: \(self.scratchBuffer.count)")
        self.recursivePrint(shape: shape, offset: 0, stride: 1)
    }
    
}

