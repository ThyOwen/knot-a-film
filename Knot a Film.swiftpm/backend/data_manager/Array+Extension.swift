//
//  Array+Extension.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 2/5/25.
//

extension Array where Element == MoviePerson {
    public func withDictionaryAccess(operation : (_ dictionary : consuming [String : MoviePerson]) -> Void) {
        
        var moviePeopleDictionary : [String : MoviePerson] = [:]
        
        for person in self {
            moviePeopleDictionary[person.name] = person
        }
        
        operation(moviePeopleDictionary)
    }
}
