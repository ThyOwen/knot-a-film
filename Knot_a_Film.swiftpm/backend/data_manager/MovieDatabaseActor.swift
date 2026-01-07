//
//  MovieNode.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 2/6/25.
//

import SwiftData
import Foundation

public enum MovieDataBaseError : Error {
    case noCSVFileFound
    case noDatabaseFileFound
}

@ModelActor
public final actor MovieDatabaseActor {
    

    public func withModelContext<T>(_ closure: (borrowing ModelContext) throws -> T) async rethrows -> T {
        try closure(self.modelContext)
    }
    
    public func fetchMovieDTOs(_ fetchDescription: consuming FetchDescriptor<Movie>) async throws -> [MovieDTO] {
        let results = try self.modelContext.fetch(fetchDescription)
        return results.map { MovieDTO(from: $0) }
    }
    
    public func fetchMovieDTOsSorted(_ fetchDescription: consuming FetchDescriptor<Movie>, sortedBy ids: [String]) async throws -> [MovieDTO] {
        let results = try self.modelContext.fetch(fetchDescription)
        let idToIndex = Dictionary(uniqueKeysWithValues: ids.enumerated().map { ($1, $0) })
        return results
            .sorted { (idToIndex[$0.rottenId] ?? Int.max) < (idToIndex[$1.rottenId] ?? Int.max) }
            .map { MovieDTO(from: $0) }
    }
    
    public func fetchMovieIDsSortedByDistance(_ fetchDescription: consuming FetchDescriptor<Movie>, distances: [(String, Double)], priorityTitle: String) async throws -> [String] {
        let results = try self.modelContext.fetch(fetchDescription)
        let distanceMap = Dictionary(uniqueKeysWithValues: distances)
        
        return zip(results, results.map { distanceMap[$0.rottenId] ?? Double.infinity })
            .sorted { leftMovie, rightMovie in
                if leftMovie.0.title.lowercased() == priorityTitle.lowercased() {
                    return true
                } else {
                    return leftMovie.1 < rightMovie.1
                }
            }
            .map { $0.0.rottenId }
    }
    
    public func fetchMoviePersonDTOs(_ fetchDescription: consuming FetchDescriptor<MoviePerson>) async throws -> [MoviePersonDTO] {
        let results = try self.modelContext.fetch(fetchDescription)
        return results.map { MoviePersonDTO(from: $0) }
    }
    
    public func updateMovieScores(_ updates: [(id: PersistentIdentifier, score: Double)]) async throws {
        for update in updates {
            if let movie = self.modelContext.model(for: update.id) as? Movie {
                movie.contentionScore = update.score
            }
        }
    }
    
    public func fetchAndPrepareMovies() async throws -> ([MovieDTO], [MoviePersonDTO]) {
        
        let movieFetch : FetchDescriptor<Movie> = .init(predicate: #Predicate { $0.dateWatched != nil })
        
        let movieResults = try modelExecutor.modelContext.fetch(movieFetch)
        movieResults.enumerated().forEach { idx, movie in
            movie.positionIndex = idx
        }
        let movies = movieResults.map { MovieDTO(from: $0) }
        
        let moviePeopleFetch = FetchDescriptor<MoviePerson>(
            predicate: #Predicate {
                $0.directedMovies.contains(where: { $0.dateWatched != nil }) ||
                $0.actedMovies.contains(where: { $0.dateWatched != nil }) ||
                $0.writtenMovies.contains(where: { $0.dateWatched != nil })
            }
        )
        
        let moviePeopleResults = try modelExecutor.modelContext.fetch(moviePeopleFetch)
        moviePeopleResults.enumerated().forEach { idx, movie in
            movie.positionIndex = idx
        }
        
        let moviePeople = moviePeopleResults.map { MoviePersonDTO(from: $0) }

        return (movies, moviePeople)
        
                
    }
    
    public func fetchAndPrepareMoviePeople(_ fetchDescription: FetchDescriptor<MoviePerson>) async throws -> [MoviePersonDTO] {
        let context = modelExecutor.modelContext
        let results = try context.fetch(fetchDescription)
        results.enumerated().forEach { idx, moviePerson in
            moviePerson.positionIndex = idx
        }
        return results.map { MoviePersonDTO(from: $0) }
    }
    
    public func withFetchResult<N, T : PersistentModel>(_ fetchDescription: consuming FetchDescriptor<T>, _ closure: (consuming [T], isolated MovieDatabaseActor) async throws -> N) async throws -> N {
        let results = try self.modelContext.fetch(fetchDescription)
        let returnValue = try await closure(results, self)
        return returnValue
    }
    
    public func withFetchResult<T : PersistentModel>(_ fetchDescription: consuming FetchDescriptor<T>, _ closure: (consuming [T], isolated MovieDatabaseActor) async throws -> Void) async throws {
        let results = try self.modelContext.fetch(fetchDescription)
        try await closure(results, self)
    }
    
    public func withMutableFetchResult<T : PersistentModel>(_ fetchDescription: consuming FetchDescriptor<T>, _ closure: (inout [T], isolated MovieDatabaseActor) async throws -> Void) async throws {
        var results = try self.modelContext.fetch(fetchDescription)
        try await closure(&results, self)
    }
    
    public static func loadModel(overwrite: Bool) async throws -> MovieDatabaseActor {
        let fileManager = FileManager.default

        let documentDirectoryURL = try fileManager.url(for: .documentDirectory,
                                                       in: .userDomainMask,
                                                       appropriateFor: nil,
                                                       create: true)
        
        let databaseURL = documentDirectoryURL.appendingPathComponent("user.store")
        
        print("Target Database URL: \(databaseURL.path)")

        if overwrite {
            if fileManager.fileExists(atPath: databaseURL.path) {
                print("Overwrite requested. Deleting existing database at: \(databaseURL.path)")
                try fileManager.removeItem(at: databaseURL)
            } else {
                print("Overwrite requested, but no existing database was found to delete.")
            }
        }

        var shouldParseCSV = false
        
        if !fileManager.fileExists(atPath: databaseURL.path) {
            print("User store not found. Attempting to seed from bundle...")
            
            if let defaultStoreURL = Bundle.main.url(forResource: "default", withExtension: "store") {
                print("Found 'default.store' in bundle. Copying to user directory...")
                try fileManager.copyItem(at: defaultStoreURL, to: databaseURL)
            } else {
                shouldParseCSV = true
            }
        } else {
            print("Existing 'user.store' found. Loading existing data.")
        }

        print("Initializing ModelContainer...")
        let modelSchema = Schema([Movie.self, MoviePerson.self])
        let modelConfiguration = ModelConfiguration(
            schema: modelSchema,
            url: databaseURL,
            allowsSave: true,
            cloudKitDatabase: .none
        )

        let modelContainer = try ModelContainer(
            for: modelSchema,
            configurations: modelConfiguration
        )
        
        if shouldParseCSV {
            print("'default.store' not found in bundle. A new empty database will be created.")
            try await DataManager.createDatabase(with: modelContainer)
        }

        return MovieDatabaseActor(modelContainer: modelContainer)
    }
}
