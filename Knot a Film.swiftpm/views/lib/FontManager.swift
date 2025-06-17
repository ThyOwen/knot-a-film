//
//  MagicFont.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 2/1/25.
//

import SwiftUI

public struct FontJSON : Codable{
    let fileName : String
    let fileExtension : String
    let name : String
}

public enum FontError : Error {
    case fontsFileJSONNotFound
    case fontsFileJSONDecodingFailed
    case failedToCreateFont(_ fontName : String)
    case failedToRegisterFont(_ fontName : String, errorDescription : String)
}

public enum ThemeFont : String {
    case jasper = "Jasper Solid (BRK)"
    case gunday = "Gunday Blur Demo"
    case quadratum = "Quadratum Unum"
    case robopixies = "RoboPixies New"
    case salisbury = "Salisbury"
    case superDream = "Super Dream"
    case worldsFinest = "WorldsFinest"
    case dailyNews = "Daily News 1915"
    case garamond = "Apple Garamond"
    case cipitillo = "Cipitillo"
    case mainframe = "Mainframe"
}

public struct FontManager {
    public static func registerFonts() {

        let fontJSON : [FontJSON] = try! Self.loadFonts()
        
        for fontJSONParameters in fontJSON {
            try? Self.registerFont(fontJSONParameters)
        }
    }
    
    private static func registerFont(_ json : consuming FontJSON) throws {
        
        guard let fontURL = Bundle.main.url(forResource: json.fileName, withExtension: json.fileExtension) else {
            throw FontError.failedToCreateFont(json.name)
        }

        var error: Unmanaged<CFError>?
        
        CTFontManagerRegisterFontsForURL(fontURL as CFURL, .process, &error)

        if let unwrappedError = error {
            let cfErrorDescription : String = unwrappedError.takeRetainedValue().localizedDescription
            throw FontError.failedToRegisterFont(json.name, errorDescription: cfErrorDescription)
        }
    }
    
    private static func loadFonts() throws -> [FontJSON] {
        guard let url = Bundle.main.url(forResource: "fonts", withExtension: "json") else {
            throw FontError.fontsFileJSONNotFound
        }
        
        let jsonData = try Data(contentsOf: url)

        let fonts = try JSONDecoder().decode([FontJSON].self, from: jsonData)
        return fonts
    }
}

