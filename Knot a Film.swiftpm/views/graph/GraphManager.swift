//
//  GraphManager.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 9/7/25.
//

import Observation
import SwiftUI
import Foundation
import SwiftData

@Observable
@MainActor public class GraphManager {

    public let renderer : GraphRenderer
    
    public let nodes : [MovieDTO]
    public let edges : [MovieEdge]
    
    public var userTranslate : UnitPoint = .center
    public var userZoom : CGFloat = 1
    public var userZoomCenter : UnitPoint = .zero
    
    public let graphParams : GraphParams = .init()
    
    public let isCircularized : Bool = false
    
    
    public required init(of watchedMovies : consuming [MovieDTO]) {

        let edges = Self.findConnections(between: watchedMovies)
        print(edges.count)
        let connectionsIndices : [SIMD2<UInt32>] = edges.map { SIMD2<UInt32>(UInt32($0.aNodePositionIndex), UInt32($0.bNodePositionIndex)) }
        
        self.nodes = consume watchedMovies
        self.edges = edges
        self.renderer = .init(connections: connectionsIndices)
 
    }
    
    public func initalizeEdgesBuffer() throws {
        let tempBuffer = [Int32](unsafeUninitializedCapacity: self.edges.count * 2) { buffer, initializedCount in
            for (idx, edge) in self.edges.enumerated() {
                buffer[idx] = Int32(edge.aNodePositionIndex)
                buffer[idx + 1] = Int32(edge.bNodePositionIndex)
            }
            initializedCount = self.edges.count * 2
        }
        let _ = tempBuffer.withUnsafeBufferPointer { bufferPointer in
            memcpy(self.renderer.connectionsBuffer.contents(), bufferPointer.baseAddress!, bufferPointer.count)
        }
    }
    
    @MainActor
    public static func create(with databaseActor : MovieDatabaseActor, using predicate : Predicate<Movie>) async throws -> Self {

        let watchedMoviesFetch = FetchDescriptor<Movie>(
            predicate: predicate
        )

        let movies = try await databaseActor.fetchAndPrepareMovies(watchedMoviesFetch)
        return Self.init(of: movies)
    }
    
    //MARK: - Connections
    private static func findConnections(between watchedMovies : borrowing [MovieDTO]) -> [MovieEdge] {
        
        var edges : [MovieEdge] = []
        
        for (aIdx, movieA) in watchedMovies.enumerated() {
            for movieB in watchedMovies[(aIdx + 1)...] { // Only consider movies after indexA
                if let edge = try? MovieEdge(movieA, movieB) {
                    edges.append(consume edge)
                }
            }
        }
        
        return edges
    }


    //MARK: - Physics
    
    private func cicularize() {
        guard self.isCircularized else {
            return
        }
        
        let radialChunkSize : Double = (2*Double.pi) / Double(self.graphParams.numNodes - 1)
        
        for idx in 0..<self.graphParams.numNodes {
            let input = Double(idx)
            
            let newX : Double = cos(input * radialChunkSize)
            let newY : Double = idx >= (self.graphParams.numNodes / 2) ? -sin(input * radialChunkSize) : sin(input * radialChunkSize)
            
            print(newX)
            print(newY)

        }
    }

}
