//
//  ViewModel.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 2/5/25.
//

import Observation
import SwiftData
import Foundation

import CoreML
import NaturalLanguage
import Accelerate

public enum ViewModelError : Error {
    case modelContextFailedToLoad
    case failedToResolveSearchModelUrl
    case failedToLoadSentenceEmbedding
    case failedToCalculateSentenceVector
    case failedToFindMovieBasedOnRottenId(rottenId : String)
    case failedToFindPromptMovie
}

@Observable
@MainActor public final class ViewModel {

    public private(set) var graph : GraphManager?
    public private(set) var recommendationEngine : RecomendationEngine?
    public private(set) var searchEngine : SearchEngine?

    public private(set) var databaseActor : MovieDatabaseActor?
    
    public var isCollapsed : Bool = true
    
    public func setup() {
        Task(priority: .background) {
            do {
                let databaseActor = try await MovieDatabaseActor.loadModel(overwrite: true)
                
                let searchEngine = try await SearchEngine.create(with: databaseActor)
                let recommendationEngine = try await RecomendationEngine.create(with: databaseActor)
                let graphManager = try await GraphManager.create(with: databaseActor, using: #Predicate<Movie> { $0.dateWatched != nil })
                        
                try await Self.testFetch(on: databaseActor)
                
                await MainActor.run {
                    self.databaseActor = consume databaseActor
                    self.recommendationEngine = recommendationEngine
                    self.searchEngine = searchEngine
                    self.graph = graphManager
                }
                
            } catch {
                print(error)
            }
        }
    }
    
    
    public func search(using likelyMisspelledUserTitle : String) {
        Task {
            guard let searchEngine, let databaseActor else {
                print("recomendation engine is nil")
                return
            }
            
            try await searchEngine.search(basedOn: likelyMisspelledUserTitle)

        }
        
    }
    
    public func recommend(using prompt : String, numSteps : Int = 2) {
        Task {
            guard let searchEngine, let graph else {
                print("recomendation engine is nil")
                return
            }
            graph.simulationTask?.cancel()
            do {

                /*
                 let movies = try await recommendationEngine.search(using: prompt) + graph.nodes
                 
                 let newGraph = GraphManager(of: movies)
                 
                 newGraph.startSimulation()
                 
                 await MainActor.run { [graph] in
                    self.graph = consume graph
                 }
                 */
            } catch {
                print(error)
            }
        }
        
    }

    public static func testFetch(on databaseActor : isolated MovieDatabaseActor) async throws {
        let desiredSet = Set(["Steven Spielberg", "Stanley Kubrick", "Alfred Hitchcock", "Martin Scorsese", "David Fincher", "Micheal Bay", "Michael Mann", "Michael Curtiz", "William Friedkin", "Paul Thomas Anderson", "Christopher Nolan", "Quinten Tarantino"])
        
        let directorFetch = FetchDescriptor<Movie>(
            predicate: #Predicate<Movie> { movie in
                movie.directors.contains(where: { desiredSet.contains($0.name)})
            }
        )
        
        
        let movies = try await databaseActor.withModelContext { modelContext in
            try modelContext.fetch(directorFetch)
        }
        
        print(movies.count)


        for idx in movies.indices {
            movies[idx].dateWatched = .now
        }

        try await databaseActor.withModelContext { modelContext in
            try modelContext.save()
        }

    }
}
