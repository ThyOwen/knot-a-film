//
//  Edge.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 1/31/25.
//

import Foundation
import SwiftData


public enum MovieEdgeError  : Error {
    case invalidNodeIndices
    case noSharedMoviePeople
}

public struct MovieEdge<N : FloatingPoint & SIMDScalar> : Identifiable {
    public let id = UUID()
    
    public let aNodePositionIndex : Int
    public let bNodePositionIndex : Int
    
    public let writers : Bool
    public let directors : Bool
    public let actors : Bool
    
    public init(_ aNode : borrowing Movie, _ bNode : borrowing Movie) throws(MovieEdgeError) {
        
        let areThereSharedWriters = Self.findSharedPeople(between: aNode.writers, and: bNode.writers)
        let areThereSharedDirectors = Self.findSharedPeople(between: aNode.directors, and: bNode.directors)
        let areThereSharedActors = Self.findSharedPeople(between: aNode.actors, and: bNode.actors)
        
        let areThereAnySharedRoles : Bool = !areThereSharedWriters && !areThereSharedDirectors && !areThereSharedActors
        
        guard !areThereAnySharedRoles else {
            throw MovieEdgeError.noSharedMoviePeople
        }
                
        guard let aIdx = aNode.positionIndex, let bIdx = bNode.positionIndex else {
            throw MovieEdgeError.invalidNodeIndices
        }
        
        self.aNodePositionIndex = aIdx
        self.bNodePositionIndex = bIdx
        
        aNode.movieConnections.append(bNode.rottenId)
        bNode.movieConnections.append(aNode.rottenId)
        
        //self.weight = Double(writer.intValue + director.intValue + performer.intValue) / 3
        
        self.writers = areThereSharedWriters
        self.directors = areThereSharedDirectors
        self.actors = areThereSharedActors
    }
    
    private static func findSharedPeople(between moviePeopleA : consuming [MoviePerson], and moviePeopleB : consuming [MoviePerson]) -> Bool {
        let moviePeopleASet : Set<String> = Set(moviePeopleA.map(\.name))
        let moviePeopleBSet : Set<String> = Set(moviePeopleB.map(\.name))
        
        let areThereSharedMoviePeople : Bool = !moviePeopleASet.intersection(moviePeopleBSet).isEmpty
        
        return areThereSharedMoviePeople
    }
}
