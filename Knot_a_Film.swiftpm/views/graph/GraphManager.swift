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

import SharedWithMetal

@Observable
@MainActor public class GraphManager {
    
    public let renderer : GraphRenderer
    
    public let nodes : [MovieDTO]
    public let edges : [MoviePersonDTO]
    
    public var userTranslate : UnitPoint = .center
    public var userZoom : CGFloat = 1
    public var userZoomCenter : UnitPoint = .zero
    
    public let isCircularized : Bool = false
    
    
    public required init(_ movies : consuming [MovieDTO],
                         _ moviePeople : consuming [MoviePersonDTO],
                         _ renderer : consuming GraphRenderer) {
        self.nodes = movies
        self.edges = moviePeople
        self.renderer = renderer
    }
    
    @MainActor public static func create(with databaseActor : MovieDatabaseActor) async throws -> Self {
        
        let (moviesTotal, moviePeople) = try await databaseActor.fetchAndPrepareMovies()
        
        let movies = Array(moviesTotal[0..<25])
        
        var flatEdges: [UInt32] = []
        var offsets: [UInt32] = [] // offsets is a per-body counts array; its length equals number of bodies.
        var numOfBodiesWithEdges: UInt32 = 0
        
        var highestNumEdges = 0
        
        for (aIdx, movieA) in movies.enumerated() {
            var edgeCount = 0
            
            for (bIdx, movieB) in movies.enumerated() {
                if bIdx == aIdx { continue }
                let hasSharedWriters = movieB.writers.contains { movieA.writers.contains($0) }
                let hasSharedDirectors = movieB.directors.contains { movieA.directors.contains($0) }
                let hasSharedActors = movieB.actors.contains { movieA.actors.contains($0) }
                
                let isEdge = hasSharedActors || hasSharedDirectors || hasSharedWriters
                
                if isEdge {
                    flatEdges.append(UInt32(bIdx))
                    edgeCount += 1
                }
            }
            
            if edgeCount > 0 {
                highestNumEdges = max(highestNumEdges, edgeCount)
                numOfBodiesWithEdges += 1
            }
            
            offsets.append(UInt32(edgeCount))
        }
        
        print("total edges", flatEdges)
        print("edge offsets", offsets)
        print("highestNumEdges", highestNumEdges)
        print("numOfBodiesWithEdges", numOfBodiesWithEdges)
        
        let renderer = GraphRenderer.init(edges: flatEdges, offsets: offsets)
        return Self.init(movies, moviePeople, renderer)
    }
    
    //MARK: - Edges
    private static func findEdges(between watchedMovies : borrowing [MovieDTO]) -> [MovieEdge] {
        
        var edges : [MovieEdge] = []
        
        print("computing edges")
        for (aIdx, aNode) in watchedMovies.enumerated() {
            for bNode in watchedMovies[(aIdx + 1)...] { // Only consider movies after indexA
                
                let areThereSharedWriters = MovieEdge.findSharedPeople(between: aNode.writers, and: bNode.writers)
                let areThereSharedDirectors = MovieEdge.findSharedPeople(between: aNode.directors, and: bNode.directors)
                let areThereSharedActors = MovieEdge.findSharedPeople(between: aNode.actors, and: bNode.actors)
                
                let areThereAnySharedRoles : Bool = !areThereSharedWriters && !areThereSharedDirectors && !areThereSharedActors
                
                guard !areThereAnySharedRoles else {
                    continue//throw MovieEdgeError.noSharedMoviePeople
                }
                
                guard let aIdx = aNode.positionIndex, let bIdx = bNode.positionIndex else {
                    continue// MovieEdgeError.invalidNodeIndices
                }
                
                let edge = MovieEdge(aNodePositionIndex: aIdx,
                                     bNodePositionIndex: bIdx,
                                     writers: areThereSharedWriters,
                                     directors: areThereSharedDirectors,
                                     actors: areThereSharedActors)
                
                edges.append(edge)
            }
        }
        return edges
    }
    
}
