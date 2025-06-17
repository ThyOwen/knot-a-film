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
    
    public func withFetchResult<N, T : PersistentModel>(_ fetchDescription: consuming FetchDescriptor<T>, _ closure: @Sendable (consuming [T]) async throws -> N) async throws -> N {
        let results = try self.modelContext.fetch(fetchDescription)
        return try await closure(results)
    }
    
    public func withFetchResult<T : PersistentModel>(_ fetchDescription: consuming FetchDescriptor<T>, _ closure: (consuming [T]) async throws -> Void) async throws {
        let results = try self.modelContext.fetch(fetchDescription)
        try await closure(results)
    }
    
    public func withMutableFetchResult<T : PersistentModel>(_ fetchDescription: consuming FetchDescriptor<T>, _ closure: (inout [T]) async throws -> Void) async throws {
        var results = try self.modelContext.fetch(fetchDescription)
        try await closure(&results)
    }
    
    public static func loadModel(overwrite : Bool) async throws -> MovieDatabaseActor {
        
        let documentDirectoryURL = try FileManager.default.url(for: .documentDirectory,
                                                               in: .userDomainMask,
                                                               appropriateFor: .applicationDirectory,
                                                               create: true)

        let databaseURL = documentDirectoryURL.appendingPathComponent("user.store")
        
        if overwrite {
            try? FileManager.default.removeItem(at: databaseURL)
        }
        
        let databaseFileExists = (try? databaseURL.checkResourceIsReachable()) ?? false

        //copy the file if it doesn't exist
        if let defaultStoreURL = Bundle.main.url(forResource: "default", withExtension: "store"), !databaseFileExists {
            try FileManager.default.copyItem(at: defaultStoreURL, to: databaseURL)
        }
        
        let modelSchema : Schema = .init([Movie.self, MoviePerson.self])

        let modelConfiguration = ModelConfiguration(schema: modelSchema, url: databaseURL, allowsSave: true, cloudKitDatabase: .none)

        let modelContainer = try ModelContainer(
            for: modelSchema,
            configurations: consume modelConfiguration
        )

        let defaultStoreExists = (try? Bundle.main.url(forResource: "default", withExtension: "store")?.checkResourceIsReachable()) ?? false
        
        if !databaseFileExists, !defaultStoreExists {
            try await DataManager.createDatabase(with: modelContainer)
            return try await self.loadModel(overwrite: false)
        } else {
            let databaseActor = MovieDatabaseActor(modelContainer: consume modelContainer)
            
            return databaseActor
        }
    }

}
