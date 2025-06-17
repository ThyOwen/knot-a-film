//
//  MovieModelTypes.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 1/17/25.
//
import Foundation
import SwiftUI

public enum Genre: String, Codable, CaseIterable {
    case actionAdventure = "Action & Adventure"
    case animation = "Animation"
    case comedy = "Comedy"
    case drama = "Drama"
    case horror = "Horror"
    case romance = "Romance"
    case scienceFictionFantasy = "Science Fiction & Fantasy"
    case mysterySuspense = "Mystery & Suspense"
    case artHouseInternational = "Art House & International"
    case classic = "Classic"
    case musicalPerformingArts = "Musical & Performing Arts"
    case western = "Western"
    case specialInterest = "Special Interest"
    case kidsFamily = "Kids & Family"
    case gayLesbian = "Gay & Lesbian"
    case sportsFitness = "Sports & Fitness"
    case documentary = "Documentary"
    case faithSpirituality = "Faith & Spirituality"
    case television = "Television"
    case cult = "Cult"
    
    public var color : Color {
        switch self {
        case .actionAdventure:
            Color.init(#colorLiteral(red: 0.8556595011, green: 0.4695341222, blue: 0.1580412938, alpha: 1))
        case .animation:
            Color.init(#colorLiteral(red: 0.5628981434, green: 0.8280586518, blue: 0.2871472116, alpha: 1))
        case .comedy:
            Color.init(#colorLiteral(red: 0.855659501, green: 0.6669908662, blue: 0.2735443116, alpha: 1))
        case .drama:
            Color.init(#colorLiteral(red: 0.2846903463, green: 0.4267473404, blue: 0.7431329618, alpha: 1))
        case .horror:
            Color.init(#colorLiteral(red: 0.8068272293, green: 0.4014994682, blue: 0.3541311489, alpha: 1))
        case .romance:
            Color.init(#colorLiteral(red: 0.8492900743, green: 0.3478415949, blue: 0.5525254997, alpha: 1))
        case .scienceFictionFantasy:
            Color.init(#colorLiteral(red: 0.4702036343, green: 0.8768909236, blue: 0.7248168181, alpha: 1))
        case .mysterySuspense:
            Color.init(#colorLiteral(red: 0.4638940943, green: 0.1629401065, blue: 0.9066149151, alpha: 1))
        case .artHouseInternational:
            Color.init(#colorLiteral(red: 0.322979487, green: 0.6725863837, blue: 0.8768909236, alpha: 1))
        case .classic:
            Color.init(#colorLiteral(red: 0.671461658, green: 0.753748673, blue: 0.7487532333, alpha: 1))
        case .musicalPerformingArts:
            Color.init(#colorLiteral(red: 0.9427083333, green: 0.33411324, blue: 0.5016511002, alpha: 1))
        case .western:
            Color.init(#colorLiteral(red: 0.8853834926, green: 0.5455727453, blue: 0.1319434164, alpha: 1))
        case .specialInterest:
            Color.init(#colorLiteral(red: 0.6030055732, green: 0.4595353696, blue: 0.5865815399, alpha: 1))
        case .kidsFamily:
            Color.init(#colorLiteral(red: 0.6285375767, green: 0.9257231953, blue: 0.821014787, alpha: 1))
        case .gayLesbian:
            Color.init(#colorLiteral(red: 0.5524444497, green: 0.315150467, blue: 0.7070395435, alpha: 1))
        case .sportsFitness:
            Color.init(#colorLiteral(red: 0.3080957546, green: 0.6030055732, blue: 0.3175486728, alpha: 1))
        case .documentary:
            Color.init(#colorLiteral(red: 0.679800238, green: 0.634658011, blue: 0.7346403928, alpha: 1))
        case .faithSpirituality:
            Color.init(#colorLiteral(red: 0.9882352941, green: 0.9189106712, blue: 0.8013279509, alpha: 1))
        case .television:
            Color.init(#colorLiteral(red: 0.3687952314, green: 0.6030055732, blue: 0.5911719549, alpha: 1))
        case .cult:
            Color.init(#colorLiteral(red: 0.2392156869, green: 0.6745098233, blue: 0.9686274529, alpha: 1))
        }
    }
    
}

public enum ContentRating: String, Codable {
    case pg = "PG"
    case g = "G"
    case r = "R"
    case pg13 = "PG-13"
    case nr = "NR"
    case none = ""
}

public enum AudienceStatus: String, Codable {
    case upright = "Upright"
    case spilled = "Spilled"
    case none = ""
}

public enum CriticsStatus: String, Codable {
    case fresh = "Fresh"
    case certifiedFresh = "Certified-Fresh"
    case rotten = "Rotten"
    case none = ""
}

public enum Studio: String, Codable, CaseIterable {
    case other
    case hulu = "Hulu"
    case twentiethCentury = "20th Century"
    case anchor = "Anchor"
    case disney = "Disney"
    case lionsgate = "Lionsgate"
    case miramax = "Miramax"
    case rko = "RKO"
    case touchStone = "Touchstone"
    case weinstein = "Weinstein"
    case a24 = "A24"
    case apple = "Apple"
    case dreamworks = "Dreamworks"
    case mgmAndUa = "MGM UA"
    case netflix = "Netflix"
    case samGoldwyn = "Samuel Goldwyn"
    case triStar = "Tri Star"
    case amazon = "Amazon"
    case bleecker = "Bleecker"
    case focus = "Focus"
    case mgm = "MGM"
    case buena = "Buena"
    case paramount = "Paramount"
    case newLine = "New Line"
    case skydance = "Skydance"
    case united = "United"
    case amblin = "Amblin"
    case columbia = "Columbia"
    case Fox = "Fox"
    case magnolia = "Magnolia"
    case orion = "Orion"
    case sony = "Sony"
    case universal = "Universal"
    case american = "American"
    case criterion = "Criterion"
    case hbo = "HBO"
    case marvel = "Marvel"
    case pixar = "Pixar"
    case summit = "Summit"
    case warner = "Warner"
    
    public static let streamers : Set<Studio> = Set([.hulu, .netflix, .hbo, .apple, .amazon])
}
