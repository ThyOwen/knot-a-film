//
//  Graph.swift
//  MetalTest
//
//  Created by Owen O'Malley on 8/17/25.
//

import MetalPerformanceShadersGraph

struct GraphParams {
    let edgeRepulsion : Double = 10.0
    let edgeAttraction : Double = 0.001
    let damping : Double = 0.4
}

@Observable final class Graph {

    let graphParams = GraphParams()
    let graph : MPSGraph = .init()
    let metalDevice : MTLDevice
    let graphDevice : MPSGraphDevice
    let numNodes : NSNumber
    
    private var positionsTensorData : MPSGraphTensorData
    private var velocitiesTensorData : MPSGraphTensorData
    
    private var outputPositions : MPSGraphTensor?
    private var outputVelocities : MPSGraphTensor?
    
    private var positionsTensor : MPSGraphTensor?
    private var velocitiesTensor : MPSGraphTensor?
    
    private var positionsBuffer : MTLBuffer
    private var velocitiesBuffer : MTLBuffer
    
    public var x : [Float32] = []
    public var y : [Float32] = []
    
    
    init(x : [Float32], y : [Float32]) {
        
        assert(x.count == y.count)
        
        let numNodes = x.count
        
        self.x = x
        self.y = y
        
        self.numNodes = NSNumber(integerLiteral: numNodes)
        let device = MTLCreateSystemDefaultDevice()!
        self.metalDevice = device
        self.graphDevice = MPSGraphDevice(mtlDevice: device)

        let positions = zip(x, y).map { SIMD2<Float32>($0.0, $0.1) }
        
        let positionsBuffer = self.metalDevice.makeBuffer(bytes: positions, length: MemoryLayout<simd_float2>.size * numNodes, options: .storageModeShared)!
        let velocitiesBuffer = self.metalDevice.makeBuffer(length: MemoryLayout<simd_float2>.size * numNodes, options: .storageModeShared)!
        
        self.positionsTensorData = MPSGraphTensorData(positionsBuffer, shape: [2, NSNumber(value: numNodes)], dataType: .float32)
        self.velocitiesTensorData = MPSGraphTensorData(velocitiesBuffer, shape: [2, NSNumber(value: numNodes)], dataType: .float32)
        
        self.positionsBuffer = positionsBuffer
        self.velocitiesBuffer = velocitiesBuffer
        
    }
    
    func buildGraph(debugPrint: Bool = true) {
        let inputNodeShape: [NSNumber] = [2, self.numNodes]

        let positionsTensor = self.graph.placeholder(shape: inputNodeShape, dataType: .float32, name: "positions") //[2, N]
        let velocityTensor = self.graph.placeholder(shape: inputNodeShape, dataType: .float32, name: "velocities") //[2, N]
        
        //the actual function
        
        let aNodePositions = self.graph.expandDims(positionsTensor, axes: [2], name: "a_node_positions_expanded") // [2, N, 1]
        let bNodePositions = self.graph.expandDims(positionsTensor, axes: [1], name: "b_node_positions_expanded") // [2, 1, N]
        
        let aNodePositionsTiled = self.graph.tileTensor(aNodePositions, withMultiplier: [1, 1, numNodes], name: "a_node_positions_tiled") // [2, N, N]
        let bNodePositionsTiled = self.graph.tileTensor(bNodePositions, withMultiplier: [1, numNodes, 1], name: "b_node_positions_tiled") // [2, N, N]
        
        //make sparse tensors
        
        let triangularIndices = Self.getTriangularIndices(length: numNodes.int32Value)

        let (aNodePositionsSparse, bNodePositionsSparse) = self.withSparseIndices(of: triangularIndices) { columnTensor, rowTensor in
            let indices = self.graph.stack([rowTensor, columnTensor], axis: 1, name: nil)
            
            let buildSparseTensor : (MPSGraphTensor) -> MPSGraphTensor = { [self] nodePositions in
                let nodePositionsSparseAxisSplit : [MPSGraphTensor] = (0..<2).map { idx in
                    
                    let start = NSNumber(value: idx)
                    let end = NSNumber(value: idx + 1)
                    
                    let nodeAxisSlice = self.graph.sliceTensor(nodePositions,
                                                               starts: [start, 0, 0],
                                                               ends: [end, numNodes, numNodes],
                                                               strides: [1,1,1], name: nil) // [1, N, N]
                    
                    let nodeAxisSliceSqueezed = self.graph.squeeze(nodeAxisSlice, axis: 0, name: nil) // [N, N]
                    
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
                
                return self.graph.stack(nodePositionsSparseAxisSplit, axis: 0, name: nil)
            }
            
            let aNodePositionsSparse : MPSGraphTensor = buildSparseTensor(aNodePositionsTiled)
            let bNodePositionsSparse : MPSGraphTensor = buildSparseTensor(bNodePositionsTiled)
            
            return (aNodePositionsSparse, bNodePositionsSparse)
        }
        
        let deltas = self.graph.subtraction(bNodePositionsTiled, aNodePositionsTiled, name: "positions_delta") // [2, N, N]
        let deltaSquared = self.graph.square(with: deltas, name: "positions_delta_squared") // [2, N, N]
        let distanceSquared = self.graph.reductionSum(with: deltaSquared, axis: 0, name: "distance_squared") // [1, N, N]
        let distances = self.graph.squareRoot(with: distanceSquared, name: "distances")// [1, N, N]
        
        let distancesTiled = self.graph.tileTensor(distances, withMultiplier: [2, 1, 1], name: "distances_tiled")// [2, N, N]
        
        let directions = self.graph.divisionNoNaN(deltas, distancesTiled, name: "directions")
        
        //calculate forces between every node and every other node
        let edgeRepulsionTensor = self.graph.constant(self.graphParams.edgeRepulsion, dataType: .float32)
        let repulsionForce = self.graph.divisionNoNaN(edgeRepulsionTensor, distanceSquared, name: "repulsion_force") // [2, N, N]
        
        let edgeIndices = Self.getRandomEdgeIndices(length: numNodes.int32Value)
        let numEdges = NSNumber(value: edgeIndices.count)
        
        let edgeAttractionTensor = self.withSparseIndices(of: edgeIndices) { columnTensor, rowTensor in
            let edgeAttractionArray = Array(repeating: Float32(self.graphParams.edgeAttraction), count: numEdges.intValue)
            
            let edgeAttractionTensor = self.buildTensor(from: edgeAttractionArray, shape: [numEdges])
            
            let edgeAttractionSparse = self.graph.sparseTensor(sparseTensorWithType: .COO,
                                                              tensors: [edgeAttractionTensor, rowTensor, columnTensor],
                                                              shape: [numNodes, numNodes],
                                                              dataType: .float32,
                                                              name: nil) // [N, N]
            return edgeAttractionSparse
        }
        
        let attractionForce = self.graph.multiplication(edgeAttractionTensor, distances, name: "attraction_force") // [2, N, N]

        let force = self.graph.subtraction(repulsionForce, attractionForce, name: "force") // [2, N, N]
        let acceleration = self.graph.multiplication(force, directions, name: "acceleration") // [2, N, N] change in velocity f = ma -> f = 1a
        
        let perNodeAcceleration = self.graph.reductionSum(with: acceleration, axis: 2, name: "per_node_acceleration") // [2, N, 1]

        let perNodeAccelerationSqueezed = self.graph.squeeze(perNodeAcceleration, axis: 2, name: "per_node_acceleration_squeezed") // [2, N]

        let updatedVelocities = self.graph.addition(velocityTensor, perNodeAccelerationSqueezed, name: "updated_velocities")// [2, N]
        
        let damping = self.graph.constant(self.graphParams.damping, dataType: .float32)
        let dampedVelocities = self.graph.multiplication(updatedVelocities, damping, name: "damped_velocities")// [2, N]
        
        let updatedPositions = self.graph.multiplication(positionsTensor, dampedVelocities, name: "updated_positions")// [2, N]
        
        let feeds : [MPSGraphTensor : MPSGraphShapedType] = [
            positionsTensor : .init(shape: [2, numNodes], dataType: .float32),
            velocityTensor : .init(shape: [2, numNodes], dataType: .float32)
        ]
        
        
        //self.graph.compile(with: self.graphDevice, feeds: feeds, targetTensors: [dampedVelocities], targetOperations: [], compilationDescriptor: MPSGraphCompilationDescriptor())
        
        self.outputVelocities = dampedVelocities
        self.outputPositions = updatedPositions
        
        self.positionsTensor = positionsTensor
        self.velocitiesTensor = velocityTensor
    }
    
    func run() {
        
        guard let outputVelocities = self.outputVelocities, let velocitiesTensor = self.velocitiesTensor else {
            fatalError("tenors not initalized")
        }
        
        guard let outputPositions = self.outputPositions, let positionsTensor = self.positionsTensor else {
            fatalError("tenors not initalized")
        }
        
        let inputFeeds : [MPSGraphTensor: MPSGraphTensorData] = [
            positionsTensor : positionsTensorData,
            velocitiesTensor : velocitiesTensorData
        ]
        
        let results = self.graph.run(
            feeds: inputFeeds,
            targetTensors: [outputVelocities, outputPositions],
            targetOperations: nil
        )

        // Extract result
        guard let resultPositions = results[outputPositions] else {
            fatalError("No result")
        }
        
        guard let resultVelocities = results[outputVelocities] else {
            fatalError("No result")
        }
                
        let outputShape = outputVelocities.shape!

        let outputLength = Self.getFlattenedSize(from: outputShape)
        var resultBuffer = [Float32](repeating: 69, count: outputLength)
        
        resultBuffer.withUnsafeMutableBytes { pointer in
            guard let address = pointer.baseAddress else {
                fatalError("base address nil")
            }
            resultPositions.mpsndarray().readBytes(address, strideBytes: nil)
        }

        // Reshape to [[x, y]]
        self.printBuffer(resultBuffer, shape: outputShape)
        
        self.x = Array(resultBuffer[0..<self.numNodes.intValue])
        self.y = Array(resultBuffer[self.numNodes.intValue..<resultBuffer.count])

        print("x", self.x)
        print("y", self.y)
        
        self.velocitiesTensorData = resultVelocities
        self.positionsTensorData = resultPositions

    }
    
    
    func withSparseIndices<T>(of indices : consuming [(Int32, Int32)], completion : (_ : borrowing MPSGraphTensor, _ : borrowing MPSGraphTensor) -> T) -> T {
  
        let length = NSNumber(value: indices.count)
        
        let columnTensor = self.buildTensor(from: indices.map(\.0), shape: [length])
        
        let rowTensor = self.buildTensor(from: indices.map(\.1), shape: [length])
        
        return completion(columnTensor, rowTensor)
        
    }
    
    func formatValue<N: Numeric>(_ value: N) -> String {
        switch value {
        case let v as Float:  return String(format: "%7.3f", v)
        case let v as Double: return String(format: "%7.3f", v)
        case let v as Int:    return String(format: "%7d", v)
        case let v as Int32:  return String(format: "%7d", v)
        case let v as Int64:  return String(format: "%7ld", v)
        default:              return "\(value)"
        }
    }

    func recursivePrint<N: Numeric>(
        _ data: [N],
        shape: [Int],
        offset: Int,
        stride: Int
    ) {
        if shape.count == 1 {
            let row = (0..<shape[0]).map { i in
                self.formatValue(data[offset + i * stride])
            }
            print(row.joined(separator: "\t"))
        } else {
            let subShape = Array(shape.dropFirst())
            let innerStride = stride * subShape.reduce(1, *)
            for i in 0..<shape[0] {
                print("Slice \(i):")
                self.recursivePrint(data, shape: subShape, offset: offset + i * innerStride, stride: stride)
                print("")
            }
        }
    }

    func printBuffer<N: Numeric>(_ data: [N], shape nsShape: [NSNumber]) {
        let shape = nsShape.map(\.intValue)
        print("Shape: \(shape)", "Data: \(data.count)")
        recursivePrint(data, shape: shape, offset: 0, stride: 1)
    }

    func buildTensor<N : Numeric>(from data : borrowing [N], shape: borrowing [NSNumber]) -> MPSGraphTensor {
        let mpsType = Self.mpsDataType(for: N.self)
        print("Using MPS type \(Self.mpsTypeToString(from: mpsType)) from Swift type \(N.self)")
        
        let outputLength = Self.getFlattenedSize(from: shape)
        
        guard outputLength == data.count else {
            fatalError("Shape \(shape) does not match data length \(data.count)")
        }
        
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
    
    @inlinable static func getNumRelevantNodes(numNodes: Int32) -> NSNumber {
        return NSNumber(value: numNodes * (numNodes - 1) / 2)
    }

    @inlinable static func mpsTypeToString(from type: MPSDataType) -> String {
        switch type {
        case .float16: return "float16"
        case .float32: return "float32"
        case .int8: return "int8"
        case .int16: return "int16"
        case .int32: return "int32"
        case .int64: return "int64"
        case .uInt8: return "uInt8"
        case .uInt16: return "uInt16"
        case .uInt32: return "uInt32"
        case .uInt64: return "uInt64"
        case .invalid: return "invalid"
        default: return "other"
        }
    }
    
    @inlinable static func mpsDataType<N: Numeric>(for type: N.Type) -> MPSDataType {
        switch type {
        case is Int8.Type:
            return .int8
        case is UInt8.Type:
            return .uInt8
        case is Int16.Type:
            return .int16
        case is UInt16.Type:
            return .uInt16
        case is Int32.Type:
            return .int32
        case is UInt32.Type:
            return .uInt32
        case is Int64.Type:
            return .int64
        case is UInt64.Type:
            return .uInt64
        case is Float32.Type:
            return .float32
        case is Float16.Type:
            return .float16
        default:
            fatalError("Unsupported numeric type in MPS type: \(type)")
        }
    }
    
    @inlinable static func getFlattenedSize(from shape : borrowing [NSNumber]) -> Int {
        return shape.reduce(1) { (currentProduct, element) in
            return currentProduct * element.intValue
        }
    }
}
