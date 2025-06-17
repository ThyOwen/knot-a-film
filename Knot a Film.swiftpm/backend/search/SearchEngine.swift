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
@Observable public final class SearchEngine {
    
    private let nlTitleEmbedding : NLEmbedding
    private let titleEmbedding : DescriptionEmbeddings

    private let databaseActor : MovieDatabaseActor
    
    public private(set) var activeSearchMovies : [Movie] = []
    
    public init(nlTitleEmbedding: NLEmbedding,
                titleEmbedding: DescriptionEmbeddings,
                databaseActor : MovieDatabaseActor) {
        self.nlTitleEmbedding = nlTitleEmbedding
        self.titleEmbedding = titleEmbedding
        
        self.databaseActor = databaseActor
    }
    
    public static func create(with databaseActor : MovieDatabaseActor) async throws -> Self {
        
        guard let titleModelPackageUrl = Bundle.main.url(forResource: "mlmodels/TitleEmbeddings", withExtension: "mlpackage") else {
            throw ViewModelError.failedToResolveSearchModelUrl
        }
        let titleModelUrl = try await MLModel.compileModel(at: titleModelPackageUrl)
        
        
        let config = MLModelConfiguration()
        
        config.computeUnits = .all
        config.allowLowPrecisionAccumulationOnGPU = true
    
        let titleEmbeddings = try DescriptionEmbeddings(contentsOf: titleModelUrl, configuration: consume config)

        let nlTitleEmbedding = try NLEmbedding.init(contentsOf: titleModelUrl)
        
        
        return Self.init(nlTitleEmbedding: nlTitleEmbedding,
                         titleEmbedding: titleEmbeddings,
                         databaseActor: databaseActor)
    }
    

    public func search(basedOn userTitle : consuming String, limit : Int = 20) async throws {
        guard let sentenceEmbedding = NLEmbedding.sentenceEmbedding(for: .english) else {
            throw ViewModelError.failedToCalculateSentenceVector
        }
        guard let vector = sentenceEmbedding.vector(for: userTitle) else {
            throw ViewModelError.failedToCalculateSentenceVector
        }
                
        let neighbors : [(String, NLDistance)] = self.nlTitleEmbedding.neighbors(for: consume vector, maximumCount: limit)
            .sorted { $0.0 < $1.0 }
        
        let searchMovieIds : Set<String> = Set(neighbors.map { $0.0 })
        
        let searchMovieDescription = FetchDescriptor<Movie>(
            predicate: #Predicate<Movie> { movie in
                searchMovieIds.contains(movie.rottenId)
            },
            sortBy: [
                .init(\Movie.title)
            ]
        )
        
        try await self.databaseActor.withFetchResult(searchMovieDescription) { movies in
            self.activeSearchMovies = zip(movies, neighbors)
                .map { ($0, $1.1) }
                .sorted { $0.1 < $1.1 }
                .map { $0.0 }

        }
        
    }
}
