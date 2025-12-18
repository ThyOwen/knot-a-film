//
//  GraphNew.swift
//  MetalTest
//
//  Created by Owen O'Malley on 8/28/25.
//

import CoreML


struct GraphNew {
    
    let graphParams : GraphParams<Float32> = .init(tileSize: 10, numTiles: 10)
    let numNodes : Int
    
    public var connections : [(Int32, Int32)] = []
    
    init(numNodes: Int) {
        self.numNodes = numNodes
    }

    mutating func buildGraphStandard() {
        withMLTensorComputePolicy(.init(.cpuAndNeuralEngine)) {
            
            let testArray : [Float16] = .init(repeating: 1.0, count: numNodes * 2)
            
            let positionTensor = MLTensor(shape: [numNodes, 2], scalars: testArray, scalarType: Float16.self) // [N, 2]
            
            let aNodePositions = positionTensor.expandingShape(at: 1).tiled(multiples: [1, numNodes, 1])
            let bNodePositions = positionTensor.expandingShape(at: 0).tiled(multiples: [numNodes, 1, 1])
            
            let deltas = aNodePositions - bNodePositions
            let distancesSquared = (deltas.squared()).sum(alongAxes: 2)
            let distances = distancesSquared.squareRoot()
            
            print(distances)
        }
    }

    static func getTriangularIndices(length : Int32) -> [(Int32, Int32)] {
        
        let size = Int(length * (length - 1)) / 2
        
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
    

}
