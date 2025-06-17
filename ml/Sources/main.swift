import TabularData
import Foundation
import CreateML
import NaturalLanguage

extension DataFrame : @unchecked @retroactive Sendable {}

extension String {
    func chunked(by chunkSize: Int) -> [String] {
        let words = self.split(separator: " ") // Split into words
        var result: [String] = []
        
        for i in stride(from: 0, to: words.count, by: chunkSize) {
            let chunk = words[i..<min(i + chunkSize, words.count)]
            result.append(chunk.joined(separator: " ")) // Join every 5 words
        }
        
        return result
    }
}

enum SearchEmbeddingsError : Error {
    case csvNotFound
    case sentenceEmbeddingNotLoaded
    case failedToCreateMLWordEmbedding
}

let headers : [String : CSVType] = [
    "rotten_tomatoes_link" : .string,
    "movie_title" : .string,
    "movie_info" : .string,
    "critics_consensus" : .string,
    "content_rating" : .string,
    "genres" : .string,
    "directors" : .string,
    "authors" : .string,
    "actors" : .string,
    "original_release_date" : .string,
    "streaming_release_date" : .string,
    "runtime" : .string,
    "production_company" : .string,
    "tomatometer_status" : .string,
    "tomatometer_rating" : .string,
    "tomatometer_count" : .string,
    "audience_status" : .string,
    "audience_rating" : .string,
    "audience_count" : .string,
    "tomatometer_top_critics_count" : .string,
    "tomatometer_fresh_critics_count" : .string,
    "tomatometer_rotten_critics_count" : .string
]

guard let url = Bundle.module.url(forResource: "rotten_tomatoes_movies", withExtension: "csv") else {
    throw SearchEmbeddingsError.csvNotFound
}

let table = try DataFrame(contentsOfCSVFile: url, types: headers)



func getEmbeddings(for key: String) async throws -> [String : [Double]] {
    let movies = await withTaskGroup(of: (String, String)?.self, returning: [(String, String)].self) { group in
        for idx in table.rows.indices {
            group.addTask {
                let rottenId = table.rows[idx]["rotten_tomatoes_link"] as? String
                let embeddingsString = table.rows[idx][key] as? String

                if let rottenId, let embeddingsString {
                    return (rottenId, embeddingsString.lowercased())
                } else {
                    return nil
                }
            }
        }

        var movies : [(String, String)] = []

        movies.reserveCapacity(table.rows.count)

        for await movie in group.compactMap(\.self) {
            movies.append(movie)
        }

        return movies
        
    }
    
    print("Creating Embeddings...")
    let embeddings = await withTaskGroup(of: (String, [Double])?.self, returning: [(String, [Double])].self) { group in
        
        //var numOfSubSentences: Int = 0
        
        for movie in movies {
            group.addTask {
                
                guard let sentenceEmbedding = NLEmbedding.sentenceEmbedding(for: .english) else {
                    return nil
                }
                
                guard let vector = sentenceEmbedding.vector(for: movie.1) else {
                    return nil
                }
                
                let rottenId = movie.0

                return (rottenId, vector)
            }
        }

        var movieEmbeddings : [(String, [Double])] = []

        for await movie in group.compactMap(\.self) {
            movieEmbeddings.append(movie)
        }
        
        
        return movieEmbeddings
        
    }

    let embeddingsDictionary : [String : [Double]] = Dictionary(uniqueKeysWithValues: embeddings)
    
    return embeddingsDictionary
}

//very important change
//let desiredSearchItem = "movie_info"
let desiredSearchItem = "movie_title"

let descriptionsEmbeddingsDictionary = try await getEmbeddings(for: desiredSearchItem)

print("Creating MLWordEmbedding...")
let wordEmbeddings = try! MLWordEmbedding(dictionary: descriptionsEmbeddingsDictionary, parameters: .init(language: .english))

print(wordEmbeddings.dimension, wordEmbeddings.vocabularySize, wordEmbeddings.description)

print("writing embeddings to file...")
try wordEmbeddings.write(toFile: "TitleEmbeddings.mlmodel")

