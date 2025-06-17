//
//  DescriptionEmbeddings.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 2/5/25.
//

import CoreML
import Accelerate

public final class DescriptionEmbeddings {
    
    static let vectorLength: Int = 512
    public let model: MLModel
    public var validMovieIds : [DescriptionEmbeddingsOutput] = []

    public init(contentsOf modelURL: URL, configuration: MLModelConfiguration) throws {
        self.model = try MLModel(contentsOf: modelURL, configuration: configuration)
    }
    
    public func computeMovieIds(comprisingOf movieIds: consuming [String]) throws {
        
        let validMovieIdEmbeddings = movieIds.map(DescriptionEmbeddingsInput.init)
        
        let outputEmbeddings = try self.predictions(inputs: consume validMovieIdEmbeddings)
        
        self.validMovieIds = outputEmbeddings
    }

    /**
        Make a prediction using the structured interface

        It uses the default function if the model has multiple functions.

        - parameters:
           - input: the input to the prediction as DescriptionEmbeddingsInput
           - options: prediction options

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as DescriptionEmbeddingsOutput
    */
    public func prediction(input: DescriptionEmbeddingsInput, options: MLPredictionOptions = MLPredictionOptions()) throws -> DescriptionEmbeddingsOutput {
        let outFeatures = try model.prediction(from: input, options: options)
        return DescriptionEmbeddingsOutput(text: input.text, features: outFeatures)
    }

    /**
        Make a batch prediction using the structured interface

        It uses the default function if the model has multiple functions.

        - parameters:
           - inputs: the inputs to the prediction as [DescriptionEmbeddingsInput]
           - options: prediction options

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as [DescriptionEmbeddingsOutput]
    */
    public func predictions(inputs: borrowing [DescriptionEmbeddingsInput], options: MLPredictionOptions = MLPredictionOptions()) throws -> [DescriptionEmbeddingsOutput] {
        let batchIn = MLArrayBatchProvider(array: inputs)
        let batchOut = try model.predictions(from: batchIn, options: options)
        var results : [DescriptionEmbeddingsOutput] = []
        results.reserveCapacity(inputs.count)
        for idx in 0..<batchOut.count {
            let outProvider = batchOut.features(at: idx)
            let result =  DescriptionEmbeddingsOutput(text: inputs[idx].text, features: outProvider)
            results.append(result)
        }
        return results
    }
    
    public func distance(between first: borrowing DescriptionEmbeddingsOutput, and second: borrowing DescriptionEmbeddingsOutput) -> Double {
        first.vector.withUnsafeBufferPointer(ofType: Double.self) { firstPointer in
            second.vector.withUnsafeBufferPointer(ofType: Double.self) { secondPointer in
                let dot = vDSP.dot(firstPointer, secondPointer)
                
                let hypotFirst = sqrt(vDSP.sumOfSquares(firstPointer))
                
                let hypotSecond = sqrt(vDSP.sumOfSquares(secondPointer))
                
                return dot / (hypotFirst * hypotSecond)
            }
        }
    }
    
    public func distance(between keyEmbedding: borrowing DescriptionEmbeddingsOutput, andAllOf arrayOfEmbeddings: borrowing [DescriptionEmbeddingsOutput]) -> [(String, Double)] {
        
        var arrayOfDistances: [Double] = []
        
        arrayOfDistances.reserveCapacity(Self.vectorLength * arrayOfEmbeddings.count)
        arrayOfEmbeddings.forEach { embedding in
            arrayOfDistances.append(self.distance(between: keyEmbedding, and: embedding))
        }
        
        
        
        let finalDistances = zip(arrayOfEmbeddings, arrayOfDistances).map { (embedding, distance) in
            return (embedding.text, distance)
        }
        
        return finalDistances
    }

    public func neighbors(near rottenId : consuming DescriptionEmbeddingsOutput, fetchLimit : Int = 10) -> [(String, Double)] {
        let distances = self.distance(between: rottenId, andAllOf: self.validMovieIds)[..<fetchLimit].map { ($0.0, $0.1) }
        
        return distances
    }
    

}

