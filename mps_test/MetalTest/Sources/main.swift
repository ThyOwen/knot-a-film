import MetalPerformanceShadersGraph

struct GraphSimulatorParams {
    let edgeRepulsion : Double = 10.0
    let edgeAttraction : Double = 0.01
    let damping : Double = 0.9
}

struct GraphSimulator {

    let graphParams = GraphSimulatorParams()
    let graph : MPSGraph = .init()
    let graphDevice : MPSGraphDevice
    
    init() {
        let device = MTLCreateSystemDefaultDevice()!
        self.graphDevice = MPSGraphDevice(mtlDevice: device)
    }
    
    func buildForceDirectedGraph(numNodes: NSNumber, nodePositions: MPSGraphTensor, debugPrint: Bool = true) -> MPSGraphTensor {
                
        let aNodePositions = self.graph.expandDims(nodePositions, axes: [2], name: "a_node_positions_expanded") // [2, N, 1]
        let bNodePositions = self.graph.expandDims(nodePositions, axes: [1], name: "b_node_positions_expanded") // [2, 1, N]
        
        let aNodePositionsTiled = self.graph.tileTensor(aNodePositions, withMultiplier: [1, 1, numNodes], name: "a_node_positions_tiled") // [2, N, N]
        let bNodePositionsTiled = self.graph.tileTensor(bNodePositions, withMultiplier: [1, numNodes, 1], name: "b_node_positions_tiled") // [2, N, N]
        
        //make sparse tensors
        
        let triangularIndices = Self.getTriangularIndices(length: numNodes.int32Value)

        let (aNodePositionsSparse, bNodePositionsSparse) = self.withSparseIndices(of: triangularIndices) { columnTensor, rowTensor in
            let indices = self.graph.stack([rowTensor, columnTensor], axis: 1, name: nil)
            
            let buildSparseTensor : (borrowing MPSGraphTensor) -> MPSGraphTensor = { nodePositions in
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
        
        let deltas = self.graph.subtraction(bNodePositionsSparse, aNodePositionsSparse, name: "positions_delta") // [2, N, N]
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

        return perNodeAccelerationSqueezed
        
    }

    func run(x: [Float32], y: [Float32]) {
        let nodeCount = x.count
        precondition(x.count == y.count)

        // Create input array of shape [N, 2]
        var positions = [Float32]()
        for i in 0..<nodeCount {
            positions.append(x[i])
            positions.append(y[i])
        }

        // Shape and input tensor
        let shape: [NSNumber] = [2, NSNumber(value: nodeCount)]
        let inputTensor = self.graph.placeholder(shape: shape, dataType: .float32, name: "positions")

        let outputTensor = self.buildForceDirectedGraph(numNodes: NSNumber(integerLiteral: nodeCount), nodePositions: inputTensor)

        let tensorData = positions.withUnsafeMutableBufferPointer { bufferPointer -> MPSGraphTensorData in
            //let dataPointer = UnsafeMutableRawPointer(bufferPointer.baseAddress!)
            //let data = Data(bytesNoCopy: dataPointer, count: bufferPointer.count, deallocator: .free)
            let data = Data(buffer: bufferPointer)
            return MPSGraphTensorData(device: graphDevice, data: data, shape: shape, dataType: .float32)
        }

        // Run graph
        
        guard let outputShape = outputTensor.shape else {
            fatalError("output shape unknown")
        }
        
        let results = self.graph.run(
            feeds: [inputTensor: tensorData],
            targetTensors: [outputTensor],
            targetOperations: nil
        )

        // Extract result
        guard let resultTensorData = results[outputTensor] else {
            fatalError("No result")
        }

        let outputLength = Self.getFlattenedSize(from: outputShape)
        var resultBuffer = [Float32](repeating: 69, count: outputLength)
        
        resultBuffer.withUnsafeMutableBytes { pointer in
            resultTensorData.mpsndarray().readBytes(pointer.baseAddress!, strideBytes: nil)
        }

        // Reshape to [[x, y]]
        self.printBuffer(resultBuffer, shape: outputShape)

    }
    
    func withSparseIndices<T>(of indices : consuming [(Int32, Int32)], completion : (_ : borrowing MPSGraphTensor, _ : borrowing MPSGraphTensor) -> T) -> T {
  
        let length = NSNumber(value: indices.count)
        
        let columnTensor = self.buildTensor(from: indices.map(\.0), shape: [length])
        
        let rowTensor = self.buildTensor(from: indices.map(\.1), shape: [length])
        
        return completion(columnTensor, rowTensor)
        
    }
    
    func printBuffer<N : Numeric>(_ data: [N], shape nsShape: borrowing [NSNumber]) {
        
        let shape = nsShape.map(\.intValue)
        
        let format : (N) -> String = { value in
            switch value {
            case let v as Float:
                return String(format: "%7.3f", v)   // width 7, 3 decimals
            case let v as Double:
                return String(format: "%7.3f", v)
            case let v as Int:
                return String(format: "%7d", v)     // width 7 for integers
            case let v as Int32:
                return String(format: "%7d", v)
            case let v as Int64:
                return String(format: "%7ld", v)
            default:
                return "\(value)"
            }
        }
        
        print("Shape: \(shape)", "Data: \(data.count)")
        
        guard (1...3).contains(shape.count) else {
            fatalError("Output shape is too large: \(shape.count)")
        }
        
        if shape.count == 1 {
            for i in 0..<shape[0] {
                print(data[i])
            }
        } else if shape.count == 2 {
            for i in 0..<shape[0] {
                var row: [String] = []
                for j in 0..<shape[1] {
                    let idx = i * shape[1] + j
                    row.append(format(data[idx]))
                }
                print(row.joined(separator: "\t"))
            }
        } else if shape.count == 3 {
            for i in 0..<shape[0] {
                print("Slice \(i):")
                for j in 0..<shape[1] {
                    var row: [String] = []
                    for k in 0..<shape[2] {
                        let idx = i * shape[1] * shape[2] + j * shape[2] + k
                        row.append(format(data[idx]))
                    }
                    print(row.joined(separator: "\t"))
                }
                print("") // empty line between slices
            }
        }
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

let simulator = GraphSimulator()

let x: [Float32] = [0.0, 4.0, 0.0, 4.0]
let y: [Float32] = [0.0, 0.0, 4.0, 4.0]

simulator.run(x: x, y: y)

