//
//  MoviePerson.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 2/6/25.
//

import SwiftData

public enum MovieRole : Codable, Sendable  {
    case writer
    case director
    case actor
}

@Model public final class MoviePerson : Hashable {
    @Attribute(.unique) public internal(set) var name : String
    
    public internal(set) var movieRoles : Set<MovieRole>
    public var numMovies : Int = 0
    
    public var writtenMovies : [Movie] = []
    public var directedMovies : [Movie] = []
    public var actedMovies : [Movie] = []
    

    public init(name: consuming Substring, movieRoles : consuming Set<MovieRole>) {
        self.name = String(name)
        self.movieRoles = movieRoles
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(self.name)
    }
}

extension MoviePerson: Equatable  {
    public static func == (lhs: borrowing MoviePerson, rhs: borrowing MoviePerson) -> Bool {
        lhs.id == rhs.id
    }
}

public struct MoviePersonDTO: Sendable, Identifiable, Hashable {
    public let id: PersistentIdentifier
    public let name: String
    public let movieRoles: Set<MovieRole>
    public let numMovies: Int
    
    public init(from model: MoviePerson) {
        self.id = model.id
        self.name = model.name
        self.movieRoles = model.movieRoles
        self.numMovies = model.numMovies
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}
