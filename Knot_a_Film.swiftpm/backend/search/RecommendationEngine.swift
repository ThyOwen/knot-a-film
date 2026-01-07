//
//  RecommendationEngine.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 6/4/25.
//

import Observation
import NaturalLanguage
import CoreML
import SwiftData

import Accelerate


//this class is responsible for both the recommendation and searching of the movie dataset
@Observable public final class RecomendationEngine {
    
    private let nlDescriptionEmbedding : NLEmbedding
    private let descriptionEmbedding : DescriptionEmbeddings

    private let databaseActor : MovieDatabaseActor
    
    public init(nlDescriptionEmbedding: NLEmbedding,
                descriptionEmbedding: DescriptionEmbeddings,
                databaseActor : MovieDatabaseActor) {
        self.nlDescriptionEmbedding = nlDescriptionEmbedding
        self.descriptionEmbedding = descriptionEmbedding
        
        self.databaseActor = databaseActor
    }
    
    public static func create(with databaseActor : MovieDatabaseActor) async throws -> Self {
        guard let descriptionModelPackageUrl = Bundle.main.url(forResource: "mlmodels/DescriptionEmbeddings", withExtension: "mlpackage") else {
            throw ViewModelError.failedToResolveSearchModelUrl
        }

        let descriptionModelUrl = try await MLModel.compileModel(at: descriptionModelPackageUrl)
        
        let config = MLModelConfiguration()
        
        config.computeUnits = .all
        config.allowLowPrecisionAccumulationOnGPU = true
        
        let descriptionEmbeddings = try DescriptionEmbeddings(contentsOf: descriptionModelUrl, configuration: config)
        
        let nlDescriptionEmbedding = try NLEmbedding.init(contentsOf: descriptionModelUrl)
        
        return Self.init(nlDescriptionEmbedding: nlDescriptionEmbedding,
                         descriptionEmbedding: descriptionEmbeddings,
                         databaseActor: databaseActor)
    }
    
    public func recommend(using promptMovieId : String, numSteps : Int = 2) async throws -> [MovieDTO] {

        let promptMovieDescription = FetchDescriptor<Movie>(
            predicate: #Predicate { $0.rottenId == promptMovieId }
        )
        
        let promptMovies = try await self.databaseActor.fetchMovieDTOs(promptMovieDescription)
        
        guard let promptMovie = promptMovies.first else {
            throw ViewModelError.failedToFindPromptMovie
        }
            

        //async let initalRoleNeighbors : [(String, NLDistance)] = try self.getMovieNeighborsBasedOnRoles(promptMovie: promptMovie)

        let initialPromptNeighbors = try self.findSimilarMovie(to: promptMovieId)

        let initialNeighbors = initialPromptNeighbors// + initalRoleNeighbors
        
        let neighbors = initialNeighbors

        let neighborsDictionary : [String: [NLDistance]] = Dictionary(grouping: neighbors, by: { $0.0 })
            .mapValues { movies in
                movies.map { movie in
                    movie.1
                }
            }

        let neighborsDescription = FetchDescriptor<Movie>(
            predicate: #Predicate {
                neighborsDictionary.keys.contains($0.rottenId)
            }
        )
        
        let promptGenres = promptMovie.genres
        let promptReleaseDate = promptMovie.originalReleaseDate
        
        var neighborDTOs = try await self.databaseActor.fetchMovieDTOs(neighborsDescription)
        
        try await self.calculateRelevanceScoresForDTOs(with: promptGenres, releaseDate: promptReleaseDate, using: neighborsDictionary, for: &neighborDTOs)
        
        neighborDTOs.sort { lhs, rhs in
            lhs.contentionScore > rhs.contentionScore
        }
        
        for movie in neighborDTOs {
            print(movie.contentionScore, movie.title)
        }
        
        return neighborDTOs
    }

    
    private func findSimilarMovie(using vector : consuming MLMultiArray, limit : Int = 10) throws -> [(String, NLDistance)] {
        let neighbors = vector.withUnsafeBufferPointer(ofType: Double.self) { ptr in
            let input = Array(ptr)
            return self.nlDescriptionEmbedding.neighbors(for: consume input, maximumCount: limit)
        }
        
        return neighbors
    }
    
    private func findSimilarMovie(basedOn description : consuming String, limit : Int = 10) throws -> [(String, NLDistance)] {
        guard let vector = self.nlDescriptionEmbedding.vector(for: description) else {
            throw ViewModelError.failedToCalculateSentenceVector
        }
        
        let neighbors = nlDescriptionEmbedding.neighbors(for: consume vector, maximumCount: limit)
        
        return neighbors
    }
    
    private func findSimilarMovie(to rottenId : consuming String, limit : Int = 10) throws -> [(String, NLDistance)] {
        
        let neighbors = self.nlDescriptionEmbedding.neighbors(for: consume rottenId, maximumCount: limit)

        return neighbors
    }
    
    
    private func getMovieNeighborsBasedOnRoles(promptMovie: borrowing Movie) async throws -> [(String, NLDistance)] {

        let directors = Set(
            promptMovie.directors
                .sorted { $0.numMovies > $1.numMovies }
                .map(\.directedMovies)
                .flatMap(\.self)
                .map(\.rottenId)
                .filter { nlDescriptionEmbedding.contains($0) }
        )
        let writers = Set(
            promptMovie.writers
                .sorted { $0.numMovies > $1.numMovies }
                .map(\.writtenMovies)
                .flatMap(\.self)
                .map(\.rottenId)
                .filter { nlDescriptionEmbedding.contains($0) }
        )
        let actors = Set(
            promptMovie.actors
                .sorted { $0.numMovies > $1.numMovies }
                .map(\.actedMovies)
                .flatMap(\.self)
                .map(\.rottenId)
                .filter { nlDescriptionEmbedding.contains($0) }
        )
        
        let moviePeopleMovies = (consume directors).union((consume writers).union(consume actors))

        let rolesFetch = FetchDescriptor<Movie>(
            predicate: #Predicate<Movie> { movie in
                moviePeopleMovies.contains(movie.rottenId)
            }
        )

        let neighborsEmbeddingsIds = try await databaseActor.fetchMovieDTOs(rolesFetch).map { DescriptionEmbeddingsInput(text: $0.rottenId) }
        
        let promptEmbedding = try descriptionEmbedding.prediction(input: .init(text: promptMovie.rottenId))
        
        let neighborsEmbeddings = try descriptionEmbedding.predictions(inputs: consume neighborsEmbeddingsIds)

        let neighborDistances = descriptionEmbedding.distance(between: consume promptEmbedding, andAllOf: consume neighborsEmbeddings)

        return neighborDistances
    }
    
    
    private func calculateRelevanceScoresForDTOs(with promptGenres: Set<Genre>, releaseDate promptReleaseDate: Date?, using neighborsDictionary: consuming [String: [NLDistance]], for movies: inout [MovieDTO]) async throws {
        
        for i in 0..<movies.count {
            guard let array = neighborsDictionary[movies[i].rottenId] else {
                continue
            }
            
            let nlpScore = 0.5 * (vDSP.sum(array) / Double(array.count))
            
            let genreScore = 0.1 * (Double(promptGenres.intersection(movies[i].genres).count) / Double(promptGenres.count))

            let viewerScore = 0.3 * (Double(movies[i].tomatoMeterRating ?? 0) + Double(movies[i].audienceRating ?? 0)) / 200

            let releaseScore : Double
            
            if let promptDate = promptReleaseDate, let movieDate = movies[i].originalReleaseDate {
                let promptYear = Calendar.current.component(.year, from: promptDate)
                let movieYear = Calendar.current.component(.year, from: movieDate)
                
                releaseScore = 0.2 * (1 - (Double(abs(promptYear - movieYear)) / 100))
            } else {
                releaseScore = 0
            }

            let finalScore = nlpScore + releaseScore + genreScore + viewerScore

            movies[i].contentionScore = finalScore
        }
    }
}
