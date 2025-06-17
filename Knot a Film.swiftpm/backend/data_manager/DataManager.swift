//
//  DataManager.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 2/5/25.
//

import SwiftData
import Foundation
import CoreData



public struct DataManager : ~Copyable {
    
    private static let regex : Regex = /(?:,"|^")(""|[\w\W]*?)(?=",|"$)|(?:,(?!")|^(?!"))([^,]*?)(?=$|,)|(\r\n|\n)/
    
    @MainActor public static func createDatabase(with modelContainer : borrowing ModelContainer) async throws {
        print("loading CSV file...")
        let csvRows : [String] = try Self.loadCSVFile(fileName: "movies_dataset")

        modelContainer.mainContext.autosaveEnabled = false
        modelContainer.mainContext.undoManager = nil
        
        let allParsedRows = await parseCSVTable(csvRows)

        let moviePeople : [MoviePerson] = await Self.getMoviePeople(allParsedRows)
        
        let movies : [Movie] = await Self.getMovies(allParsedRows, moviePeople: moviePeople)
        
        for movie in movies {
            movie.updateMoviePeople()
        }
        

        try modelContainer.mainContext.transaction {

            for moviePerson in moviePeople {
                modelContainer.mainContext.insert(moviePerson)
            }
            
            for movie in movies {
                modelContainer.mainContext.insert(movie)
            }
            try modelContainer.mainContext.save()
        }
            
        
        print("Done!")
        
    }

    private static func loadCSVFile(fileName : consuming String) throws -> [String] {
        guard let url = Bundle.main.path(forResource: fileName, ofType: "csv") else {
            throw MovieDataBaseError.noCSVFileFound
        }
        
        let content = try String(contentsOfFile: url, encoding: .utf8)
        let csvRows = content.components(separatedBy: "\n")
        
        return csvRows
    }
    
    private static func regexParseCSVRow(row input : borrowing String) throws -> [Substring] {
        
        var components : [Substring] = []
        components.reserveCapacity(Movie.columnCount)
        
        for match in input.matches(of: Self.regex) {
            if let quoted = match.output.1 {
                components.append(quoted)
            } else if let unquoted = match.output.2 {
                components.append(unquoted)
            }
        }
        
        guard components.count == Movie.columnCount else {
            throw MovieParseError.missingRequiredFields
        }
        
        return components
    }
    
    private static func parseCSVTable(_ csvRows : consuming [String]) async -> [[Substring]] {
        await withTaskGroup(of: [Substring]?.self, returning: [[Substring]].self) { group in
            
            var numOfSuccessfullyParsedRows : Int = 0
            
            csvRows.forEach { csvRow in
                group.addTask {
                    if let parsedRow = try? Self.regexParseCSVRow(row: csvRow) {
                        numOfSuccessfullyParsedRows += 1
                        return parsedRow
                    } else {
                        return nil
                    }
                }
            }
            
            var allParsedRows : [[Substring]] = []
            
            for await parsedRow in group.compactMap( \.self ) {
                allParsedRows.append(consume parsedRow)
            }
            
            return allParsedRows
        }
    }
    
    private static func getMoviePeople(_ parsedRows : borrowing [[Substring]]) async -> [MoviePerson] {
        let peopleArray = await withTaskGroup(of: [(Substring, MovieRole)].self, returning: [(Substring, MovieRole)].self) { group in
            
            var numOfSuccessfullyParsedMoviePeople : Int = 0
            
            parsedRows.forEach { parsedRow in
                group.addTask {
                    async let writerArray = parsedRow[7].split(separator: ",").map { ($0, MovieRole.writer) }
                    async let directorArray =  parsedRow[6].split(separator: ",").map { ($0, MovieRole.director) }
                    async let actorArray =  parsedRow[8].split(separator: ",").map { ($0, MovieRole.actor) }
                    
                    let peopleArray = await (writerArray + actorArray + directorArray)
                    
                    numOfSuccessfullyParsedMoviePeople += peopleArray.count
                    return peopleArray
                }
            }
            
            var moviePeople : [(Substring, MovieRole)] = []
            moviePeople.reserveCapacity(numOfSuccessfullyParsedMoviePeople)
            
            for await person in group {
                moviePeople += consume person
            }
            
            return moviePeople
        }
        
        let groupedRoles = Dictionary(grouping: peopleArray, by: { $0.0 })
            .mapValues { Set($0.map { $0.1 }) }
        
        let moviePeopleArray = groupedRoles.map { name, groups in
            MoviePerson(name: name, movieRoles: groups)
        }
        
        return moviePeopleArray
    }
    /*
    private static func getMoviePeople(_ parsedRows : borrowing [[Substring]], using viewContext : inout NSManagedObjectContext) async {
        let peopleArray = await withTaskGroup(of: [(Substring, MovieRole)].self, returning: [(Substring, MovieRole)].self) { group in
            
            var numOfSuccessfullyParsedMoviePeople : Int = 0
            
            parsedRows.forEach { parsedRow in
                group.addTask {
                    async let writerArray = parsedRow[7].split(separator: ",").map { ($0, MovieRole.writer) }
                    async let directorArray =  parsedRow[6].split(separator: ",").map { ($0, MovieRole.director) }
                    async let actorArray =  parsedRow[8].split(separator: ",").map { ($0, MovieRole.actor) }
                    
                    let peopleArray = await (writerArray + actorArray + directorArray)
                    
                    numOfSuccessfullyParsedMoviePeople += peopleArray.count
                    return peopleArray
                }
            }
            
            var moviePeople : [(Substring, MovieRole)] = []
            moviePeople.reserveCapacity(numOfSuccessfullyParsedMoviePeople)
            
            for await person in group {
                moviePeople += consume person
            }
            
            return moviePeople
        }
        
        let groupedRoles = Dictionary(grouping: peopleArray, by: { $0.0 })
            .mapValues { Set($0.map { $0.1 }) }
        
        groupedRoles.forEach { name, groups in
            let moviePerson = MoviePerson(backingData: viewContext)
            moviePerson.name = name
            moviePerson.movieRoles = groups
        }
        
        let moviePeopleArray = groupedRoles.map { name, groups in
            MoviePerson(name: name, movieRoles: groups)
        }

    }
    */
    private static func getMovies(_ parsedRows : consuming [[Substring]], moviePeople : borrowing [MoviePerson]) async -> [Movie] {
        await withTaskGroup(of: Movie?.self, returning: [Movie].self) { group in
            
            var numOfSuccessfullyParsedMovies : Int = 0
            
            moviePeople.withDictionaryAccess { moviePeopleDictionary in
                parsedRows.forEach { parsedRow in
                    group.addTask {
                        if let movie = try? Movie(parsedCSVRow: parsedRow, moviePeople: moviePeopleDictionary) {
                            numOfSuccessfullyParsedMovies += 1
                            return movie
                        } else {
                            return nil
                        }
                    }
                }
            }
            
            var movies : [Movie] = []
            movies.reserveCapacity(numOfSuccessfullyParsedMovies)
            
            for await movie in group.compactMap( \.self ) {
                movies.append(consume movie)
            }
            
            return movies
        }
    }
    
}
