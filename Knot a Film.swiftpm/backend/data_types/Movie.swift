//
//  DataModel.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 1/17/25.
//

import SwiftData
import Foundation

public enum MovieParseError : Error {
    case missingRequiredFields
}

@Model public class Movie {
    @Transient public var contentionScore : Double = 0
    @Transient public var positionIndex : Int? = nil
    @Transient public var movieConnections : [String] = []
    
    public var dateWatched : Date? = nil
    
    @Attribute(.unique) public private(set) var rottenId: String

    public private(set) var title : String
    public private(set) var info : String
    public private(set) var criticsConsensus : String
    public private(set) var contentRating : ContentRating
    public private(set) var genres : Set<Genre>
    
    @Relationship(deleteRule: .cascade, inverse: \MoviePerson.writtenMovies) public var writers : [MoviePerson]
    @Relationship(deleteRule: .cascade, inverse: \MoviePerson.directedMovies) public var directors : [MoviePerson]
    @Relationship(deleteRule: .cascade, inverse: \MoviePerson.actedMovies) public var actors : [MoviePerson]
    
    public private(set) var originalReleaseDate : Date?
    public private(set) var streamingReleaseDate : Date?
    public private(set) var runtime : Int?
    public private(set) var productionCompany : String
    public private(set) var studio : Studio
    
    public private(set) var tomatoMeterStatus : CriticsStatus
    public private(set) var tomatoMeterRating : Int?
    public private(set) var tomatoMeterCount : Int?
    
    public private(set) var audienceStatus : AudienceStatus
    public private(set) var audienceRating : Int?
    public private(set) var audienceCount : Int?
    
    public private(set) var tomatoMeterTopCriticsCount : Int?
    public private(set) var tomatoMeterFreshCriticsCount : Int?
    public private(set) var tomatoMeterRottenCriticsCount : Int?
    
    public private(set) var colorsString : String
    
    public static let columnCount : Int = 23
    

    private static let dateFormatter : DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "MM-dd-yy"
        return formatter
    }()
    
    public init(parsedCSVRow: consuming [Substring], moviePeople : borrowing [String : MoviePerson]) throws {
        
        self.rottenId = String(parsedCSVRow[0])
        self.title = String(parsedCSVRow[1])
        self.info = String(parsedCSVRow[2])
        
        self.criticsConsensus = String(parsedCSVRow[3])
        self.contentRating = ContentRating(rawValue: String(parsedCSVRow[4])) ?? .none
        self.genres = Set(parsedCSVRow[5].components(separatedBy: ", ").compactMap { Genre(rawValue: $0) })

        let writerStrings = parsedCSVRow[7].components(separatedBy: ", ")
        let directorStrings = parsedCSVRow[6].components(separatedBy: ", ")
        let actorStrings = parsedCSVRow[8].components(separatedBy: ", ")
 
        self.writers = Self.findMoviePeople(writerStrings, in: moviePeople, ofType: .writer)
        self.directors = Self.findMoviePeople(directorStrings, in: moviePeople, ofType: .director)
        self.actors = Self.findMoviePeople(actorStrings, in: moviePeople, ofType: .actor)
        
        self.originalReleaseDate = Self.dateFormatter.date(from: String(parsedCSVRow[9]))
        self.streamingReleaseDate = Self.dateFormatter.date(from: String(parsedCSVRow[10]))
        
        self.runtime = Int(parsedCSVRow[11])
        let productionCompany = String(parsedCSVRow[12])

        self.productionCompany = productionCompany
        self.studio = Self.getStudioName(from: productionCompany)
        
        self.tomatoMeterStatus = CriticsStatus(rawValue: String(parsedCSVRow[13])) ?? .none
        
        self.colorsString = String(parsedCSVRow[14])
        
        self.tomatoMeterRating = Int(parsedCSVRow[15])
        self.tomatoMeterCount = Int(parsedCSVRow[16])
        
        self.audienceStatus = AudienceStatus(rawValue: String(parsedCSVRow[17])) ?? .none
        self.audienceRating = Int(parsedCSVRow[18])
        self.audienceCount = Int(parsedCSVRow[19])
        
        self.tomatoMeterTopCriticsCount = Int(parsedCSVRow[20])
        self.tomatoMeterFreshCriticsCount = Int(parsedCSVRow[21])
        self.tomatoMeterRottenCriticsCount = Int(parsedCSVRow[22])
    }
    
    private static func findMoviePeople(_ arrayOfNames : consuming [String],
                                        in moviePeopleDictionary : borrowing [String : MoviePerson],
                                        ofType role : consuming MovieRole) -> [MoviePerson] {
        var moviePeople : [MoviePerson] = []
        
        moviePeople.reserveCapacity(arrayOfNames.count)
        
        for name in arrayOfNames {
            if let person = moviePeopleDictionary[name] {
                moviePeople.append(person)
            }
        }

        return moviePeople
    }
    
    private static func getStudioName(from studioString : consuming String) -> Studio {
        var finalStudio : Studio = .other

        let studioComponets = Set(studioString.split(separator: " "))
        
        for studio in Studio.allCases {

            var numMatches : Int = 0

            for componet in studioComponets {
                if studio.rawValue.lowercased().contains(componet.lowercased()) {
                    numMatches += 1
                }
            }
            
            if numMatches != 0 {
                finalStudio = studio
            }
        }
        return finalStudio
    }

    public func updateMoviePeople() {
        for actor in self.actors {
            actor.numMovies += 1
        }
        
        for director in self.directors {
            director.numMovies += 1
        }
        
        for writer in self.writers {
            writer.numMovies += 1
        }
    }
    
    
}

extension Movie: Equatable {
    public static func == (lhs: borrowing Movie, rhs: borrowing Movie) -> Bool {
        lhs.id == rhs.id
    }
}


