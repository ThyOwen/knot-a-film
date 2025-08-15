import MetalPerformanceShadersGraph

struct GraphSimulatorParams {
    let edgeRepulsion : Double = 10.0
    let edgeAttraction : Double = 0.01
    let damping : Double = 0.9
}

struct GraphSimulator {

    let graphParams = GraphSimulatorParams()
    let graph = MPSGraph()
    
    func buildForceDirectedGraph(numNodes: NSNumber, nodePositions: MPSGraphTensor, debugPrint: Bool = true) -> MPSGraphTensor {
        
        let aNodePositions = self.graph.expandDims(nodePositions, axes: [2], name: "a_node_positions_expanded") // [2, N, 1]
        let bNodePositions = self.graph.expandDims(nodePositions, axes: [1], name: "b_node_positions_expanded") // [2, 1, N]
        
        let aNodePositionsTiled = self.graph.tileTensor(aNodePositions, withMultiplier: [1, 1, numNodes], name: "a_node_positions_tiled") // [2, N, N]
        let bNodePositionsTiled = self.graph.tileTensor(bNodePositions, withMultiplier: [1, numNodes, 1], name: "b_node_positions_tiled") // [2, N, N]
        
        //make sparse tensors
        
        let deltas = self.graph.subtraction(bNodePositionsTiled, aNodePositionsTiled, name: "positions_delta") // [2, N, N]
        let deltaSquared = self.graph.square(with: deltas, name: "positions_delta_squared") // [2, N, N]
        let distanceSquared = self.graph.reductionSum(with: deltaSquared, axis: 0, name: "distance_squared") // [1, N, N]
        let distances = self.graph.squareRoot(with: distanceSquared, name: "distances")// [1, N, N]
        
        let distancesTiled = self.graph.tileTensor(distances, withMultiplier: [2, 1, 1], name: "distances_tiled")// [2, N, N]
        
        let directions = self.graph.divisionNoNaN(deltas, distancesTiled, name: "directions")
        
        //calculate forces between every node and every other node
        let edgeRepulsionTensor = self.graph.constant(self.graphParams.edgeRepulsion, dataType: .float32)
        let repulsionForce = self.graph.divisionNoNaN(edgeRepulsionTensor, distanceSquared, name: "repulsion_force") // [2, N, N]
        
        let edgeAttractionTensor = self.graph.constant(self.graphParams.edgeAttraction, dataType: .float32)
        //select distances in the indices that have connections and create a sparse tensor.
        //must reduce the relavent distances into a 1d tensor for sparsity
        let attractionForce = self.graph.multiplication(edgeAttractionTensor, distances, name: "attraction_force") // [2, N, N]

        let force = self.graph.subtraction(repulsionForce, attractionForce, name: "force") // [2, N, N]
        let acceleration = self.graph.multiplication(force, directions, name: "acceleration") // [2, N, N] change in velocity f = ma -> f = 1a
        
        let perNodeAcceleration = self.graph.reductionSum(with: acceleration, axis: 2, name: "per_node_acceleration") // [2, N , 1]

        let perNodeAccelerationSqueezed = self.graph.squeeze(perNodeAcceleration, axis: 2, name: "per_node_acceleration_squeezed") // [2, N]
        
        //let perNodeAccelerationFlattened = self.graph.flatten2D(perNodeAccelerationSqueezed, axis: 1, name: "per_node_acceleration_flattened") // [2N]
        
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

        // Build computation graph
        let outputTensor = self.buildForceDirectedGraph(numNodes: NSNumber(integerLiteral: nodeCount), nodePositions: inputTensor)

        // Wrap Swift array into MPSGraphTensorData
        let device = MTLCreateSystemDefaultDevice()!
        let graphDevice = MPSGraphDevice(mtlDevice: device)
        
        let tensorData = positions.withUnsafeBufferPointer { buffer -> MPSGraphTensorData in
            return MPSGraphTensorData(device: graphDevice, data: Data(buffer: buffer), shape: shape, dataType: .float32)
        }

        // Run graph
        
        guard let outputShape = outputTensor.shape?.map(\.intValue) else {
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

        let outputLength = outputShape.reduce(1) { (currentProduct, element) in
            return currentProduct * element
        }

        var resultBuffer = [Float32](repeating: 69, count: outputLength)
        
        resultBuffer.withUnsafeMutableBytes { pointer in
            resultTensorData.mpsndarray().readBytes(pointer.baseAddress!, strideBytes: nil)
        }

        // Reshape to [[x, y]]

        self.printBuffer(resultBuffer, shape: outputShape)

    }
    
    func buildSparseMatrix() -> MPSGraphTensor {
        
        let indices : [(Int, Int)] = Self.getTriangularIndices(length: 3)
        
        indices.map(\.0)
        
        let mps
        
        self.graph.sparseTensor(sparseTensorWithType: .COO, tensors: [dataTensor, colTensor, rowTensor], shape: <#T##[NSNumber]#>, dataType: .float32, name: nil)
    }
    
    
    func printBuffer(_ data: [Float32], shape: [Int]) {
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
                    row.append(String(format: "%.3f", data[idx]))
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
                        row.append(String(format: "%.3f", data[idx]))
                    }
                    print(row.joined(separator: "\t"))
                }
                print("") // empty line between slices
            }
        }
    }
    
    static func getTriangularIndices(length : Int) -> [(Int, Int)] {
        
        let size = length * (length - 1) / 2
        
        var indices : [(Int, Int)] = .init()
        indices.reserveCapacity(size)
        
        for aIdx in 0..<length {
            for bIdx in (aIdx + 1)..<length {
                indices.append((aIdx, bIdx))
            }
        }
        
        return indices
    }

}

let simulator = GraphSimulator()

let x: [Float32] = [1.0, 2.0, 3.0, 4.0]
let y: [Float32] = [5.0, 6.0, 7.0, 8.0]

simulator.run(x: x, y: y)

