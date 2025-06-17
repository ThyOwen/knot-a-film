//
//  DescriptionEmbeddingsInputs.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 2/22/25.
//

import CoreML

public final class DescriptionEmbeddingsInput : MLFeatureProvider {
    public var text: String

    public let featureNames: Set<String> = ["text"]

    public func featureValue(for featureName: consuming String) -> MLFeatureValue? {
        if featureName == "text" {
            return MLFeatureValue(string: text)
        }
        return nil
    }

    public init(text: consuming String) {
        self.text = text
    }

}

public final class DescriptionEmbeddingsVectorInput : MLFeatureProvider {
    public var vector: MLMultiArray

    public let featureNames: Set<String> = ["vector"]

    public func featureValue(for featureName: String) -> MLFeatureValue? {
        if featureName == "vector" {
            return MLFeatureValue(multiArray: vector)
        }
        return nil
    }

    public init(vector: MLMultiArray) {
        self.vector = vector
    }
}

public final class DescriptionEmbeddingsOutput : MLFeatureProvider {

    /// Source provided by CoreML
    private let provider: MLFeatureProvider
    
    /// The input text (manually stored)
    public let text: String

    /// vector as multidimensional array of doubles
    public var vector: MLMultiArray {
        self.provider.featureValue(for: "vector")!.multiArrayValue!
    }

    public var vectorShapedArray: MLShapedArray<Double> {
        MLShapedArray<Double>(self.vector)
    }

    public var featureNames: Set<String> {
        var names = self.provider.featureNames
        names.insert("text") // Add "text" to available feature names
        return names
    }

    public func featureValue(for featureName: String) -> MLFeatureValue? {
        if featureName == "text" {
            return MLFeatureValue(string: text)
        }
        return self.provider.featureValue(for: featureName)
    }

    /// Initialize with vector and manually store input text
    public init(text: consuming String, vector: consuming MLMultiArray) {
        self.text = text
        self.provider = try! MLDictionaryFeatureProvider(dictionary: ["vector" : MLFeatureValue(multiArray: vector)])
    }

    public init(text: String, features: MLFeatureProvider) {
        self.text = text
        self.provider = features
    }
}


