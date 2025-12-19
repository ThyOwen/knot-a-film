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

public struct MovieEdge {
    
    public let aNodePositionIndex : Int
    public let bNodePositionIndex : Int
    
    public let writers : Bool
    public let directors : Bool
    public let actors : Bool
    
    public init (aNodePositionIndex : Int, bNodePositionIndex : Int, writers : Bool, directors : Bool, actors : Bool) {
        self.aNodePositionIndex = aNodePositionIndex
        self.bNodePositionIndex = bNodePositionIndex
        self.writers = writers
        self.directors = directors
        self.actors = actors
    }
    
    static func findSharedPeople(between moviePeopleA : [MoviePersonDTO], and moviePeopleB : [MoviePersonDTO]) -> Bool {
        for moviePersonA in moviePeopleA {
            for moviePersonB in moviePeopleB { // Only consider movies after indexA
                if moviePersonA.id == moviePersonB.id {
                    return true
                }
            }
        }
                
        return false
    }
}
