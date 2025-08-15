//
//  GraphOld.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 1/30/25.
//

import Foundation
import Observation
import SwiftUICore
import SwiftData


@Observable
@MainActor public final class GraphManager {

    public typealias SIMDType = SIMD64<Double>

    private let batchCount : Int
    private let numPoints : Int
    private static let batchSize : Int = SIMDType.scalarCount
    
    public var isCircularized : Bool = false

    public let nodes : [Movie]
    public let edges : [MovieEdge<Double>]
    
    public var positionsX : ContiguousArray<Double>
    public var positionsY : ContiguousArray<Double>
    @ObservationIgnored public private(set) var unNormalizedPositionsX : ContiguousArray<Double>
    @ObservationIgnored public private(set) var unNormalizedPositionsY : ContiguousArray<Double>
    @ObservationIgnored private var velocitiesX : ContiguousArray<Double>
    @ObservationIgnored private var velocitiesY : ContiguousArray<Double>
    
    public var activeBounds : CGSize
    public static let passiveBound : Double = 100
    
    public var userTranslate : UnitPoint = .center
    public var userZoom : CGFloat = 1
    public var userZoomCenter : UnitPoint = .zero
    
    private let repulsionStrength : Double = 1000
    private let attractionStrength : Double = 0.001
    private let damping : Double = 0.85
    
    public private(set) var simulationTask : Task<Void, Never>? = nil
    public private(set) var visualizeTask : Task<Void, Never>? = nil
    
    public init(of watchedMovies : consuming [Movie]) {
        
        self.batchCount = Int(ceil(Double(watchedMovies.count * 2) / Double(Self.batchSize)))
        
        let numOfPoints : Int = self.batchCount * Self.batchSize

        let arrayX : ContiguousArray<Double> = .init(unsafeUninitializedCapacity: numOfPoints) { buffer, initilizedCount in
            for idx in 0..<numOfPoints {
                let value = Double.random(in: -Self.passiveBound...Self.passiveBound)
                buffer[idx] = value
            }
            initilizedCount = numOfPoints
        }
        
        let arrayY : ContiguousArray<Double> = .init(unsafeUninitializedCapacity: numOfPoints) { buffer, initilizedCount in
            for idx in 0..<numOfPoints {
                let value = Double.random(in: -Self.passiveBound...Self.passiveBound)
                buffer[idx] = value
            }
            initilizedCount = numOfPoints
        }

        self.velocitiesX = .init(repeating: 0, count: numOfPoints)
        self.velocitiesY = .init(repeating: 0, count: numOfPoints)

        self.numPoints = numOfPoints
        self.edges = Self.findConnections(between: watchedMovies)

        
        self.positionsX = arrayX
        self.positionsY = arrayY
        self.unNormalizedPositionsX = consume arrayX
        self.unNormalizedPositionsY = consume arrayY
        
        self.activeBounds = .init(width: CGFloat(Self.passiveBound), height: CGFloat(Self.passiveBound))
        
        
        self.nodes = consume watchedMovies
    }
    
    public static func create(with databaseActor : borrowing MovieDatabaseActor, using predicate : Predicate<Movie>) async throws -> Self {

        let watchedMoviesFetch = FetchDescriptor<Movie>(
            predicate: predicate
        )

        let graph = try await databaseActor.withFetchResult(watchedMoviesFetch) { watchedMovies in
            watchedMovies.enumerated().forEach { idx, movie in
                movie.positionIndex = idx
            }
            return await Self.init(of: watchedMovies)
        }
        
        graph.startSimulation()
        
        return graph

    }
    
    //MARK: - Connections
    private static func findConnections(between watchedMovies : borrowing [Movie]) -> [MovieEdge<Double>] {
        
        var edges : [MovieEdge<Double>] = []
        
        for (aIdx, movieA) in watchedMovies.enumerated() {
            for movieB in watchedMovies[(aIdx + 1)...] { // Only consider movies after indexA
                if let edge = try? MovieEdge<Double>(movieA, movieB) {
                    edges.append(consume edge)
                }
            }
        }
        
        return edges
    }
    

    //MARK: - Physics
    public func startSimulation() {
        
        self.visualizeTask = Task.detached(priority: .high) {
            while true {
                await self.visualizeTick()
                try? await Task.sleep(for: .seconds(0.01))
            }
        }
        
        if !self.isCircularized {
            self.simulationTask = Task.detached(priority: .high) {
                while true {
                    //let start = Date.now
                    await self.simulateTick()
                    //print(Date.now.timeIntervalSince1970 - start.timeIntervalSince1970)
                    try? await Task.sleep(for: .seconds(0.02))
                }
            }
        } else {
            self.cicularize()
        }
    }
    
    private func simulateTick() async {
        var newVelocitiesX = self.velocitiesX
        var newVelocitiesY = self.velocitiesY
        
        var newPositionsX = self.unNormalizedPositionsX
        var newPositionsY = self.unNormalizedPositionsY

        // Apply repulsion between all nodes
        var deltaX : SIMDType = .init(repeating: 0.0)
        var deltaY : SIMDType = .init(repeating: 0.0)
        var distance : SIMDType = .init(repeating: 0.0)
        var force : SIMDType = .init(repeating: 0.0)
        
        var forceX : SIMDType = .init(repeating: 0.0)
        var forceY : SIMDType = .init(repeating: 0.0)
        /*
        for idx in self.numPoints..<(Self.batchSize * self.batchCount) {
            newPositions[idx] = N.nan
            newVelocities[idx] = N.nan
        }
        */
        for aBatchIdx in 0..<self.batchCount {
            for bBatchIdx in (aBatchIdx + 1)..<self.batchCount {
                
                let aRange = Self.batchSize*aBatchIdx..<Self.batchSize*(aBatchIdx+1)
                let bRange = Self.batchSize*bBatchIdx..<Self.batchSize*(bBatchIdx+1)
                
                Self.withRangedSIMDAccess(to: newPositionsX, aRange: aRange, bRange: bRange) { aNodePositionsX, bNodePositionsX in
                    Self.withRangedSIMDAccess(to: newPositionsY, aRange: aRange, bRange: bRange) { aNodePositionsY, bNodePositionsY in
                        
                        deltaX = bNodePositionsX - aNodePositionsX
                        deltaY = bNodePositionsY - aNodePositionsY
                        
                        distance = (deltaX * deltaX + deltaY * deltaY).squareRoot()
                        //let distanceMask = distance .< 1.0
                        //distance.replace(with: 1.0, where: distanceMask)
                        
                        force = self.repulsionStrength / (distance * distance)
                        
                        deltaX /= distance
                        deltaY /= distance
                        
                        forceX = force * deltaX
                        forceY = force * deltaY
                        
                        Self.withRangedSIMDAccess(to: &newVelocitiesX, aRange: aRange, bRange: bRange) { aNodeVelocitiesX, bNodeVelocitiesX in
                            Self.withRangedSIMDAccess(to: &newVelocitiesY, aRange: aRange, bRange: bRange) { aNodeVelocitiesY, bNodeVelocitiesY in
                                aNodeVelocitiesX -= forceX
                                aNodeVelocitiesY -= forceY
                                
                                bNodeVelocitiesX += forceX
                                bNodeVelocitiesY += forceY
                            }
                        }
                    }
                }
                
            }
        }

        // Apply attraction along edges
        for edge in self.edges {

            edge.withBufferAccess(to: newPositionsX) { aNodePositionX, bNodePositionX in
                edge.withBufferAccess(to: newPositionsY) { aNodePositionY, bNodePositionY in
                    var deltaX = bNodePositionX - aNodePositionX
                    var deltaY = bNodePositionY - aNodePositionY
                    
                    let distance = hypot(deltaX, deltaY)
                    
                    let edgeStrength : Double = Double((edge.directors ? 1.0 : 0.0) +
                                             (edge.actors ? 1.0 : 0.0) +
                                             (edge.writers ? 1.0 : 0.0)) * self.attractionStrength
                    
                    let force = distance * edgeStrength
                    
                    deltaX /= distance
                    deltaY /= distance
                    
                    edge.withBufferAccess(to: &newVelocitiesX) { aNodeVelocityX, bNodeVelocityX in
                        edge.withBufferAccess(to: &newVelocitiesY) { aNodeVelocityY, bNodeVelocityY in
                            aNodeVelocityX += force * deltaX
                            aNodeVelocityY += force * deltaY
                            
                            bNodeVelocityX -= force * deltaX
                            bNodeVelocityY -= force * deltaY
                        }
                    }
                    
                }
            }
        }
        
        // Update positions and apply damping
        for batchIdx in 0..<self.batchCount {
            let bounds: Range<Int> = Self.batchSize*batchIdx..<Self.batchSize*(batchIdx+1)
            
            Self.withSIMDAccess(to: &newPositionsX[bounds]) { positionX in
                Self.withSIMDAccess(to: &newPositionsY[bounds]) { positionY in
                    Self.withSIMDAccess(to: &newVelocitiesX[bounds]) { velocityX in
                        Self.withSIMDAccess(to: &newVelocitiesY[bounds]) { velocityY in
                            velocityX *= self.damping
                            velocityY *= self.damping
                            
                            positionX += velocityX
                            positionY += velocityY
                        }
                    }
                }
            }
        }

        await MainActor.run { [newPositionsX, newPositionsY] in
            self.unNormalizedPositionsX = consume newPositionsX
            self.unNormalizedPositionsY = consume newPositionsY
        }
        
        self.velocitiesX = consume newVelocitiesX
        self.velocitiesY = consume newVelocitiesY
    }
    

    private func visualizeTick() async {
        var normalizedPositionsX = ContiguousArray<Double>.init(repeating: 0.0, count: self.numPoints)
        var normalizedPositionsY = ContiguousArray<Double>.init(repeating: 0.0, count: self.numPoints)

        var maxX: Double = 0
        var maxY: Double = 0
        var minX: Double = 0
        var minY: Double = 0

        var averageX: Double = 0
        var averageY: Double = 0

        for batchIdx in 0..<self.batchCount {
            let bounds: Range<Int> = (Self.batchSize * batchIdx)..<(Self.batchSize * (batchIdx + 1))

            Self.withSIMDAccess(to: self.unNormalizedPositionsX[bounds]) { positionSIMD in
                maxX = max(maxX, positionSIMD.max())
                minX = min(minX, positionSIMD.min())

                averageX += positionSIMD.sum()

                // Copy positions to normalized array
                Self.withSIMDAccess(to: &normalizedPositionsX[bounds]) { normalizedPositionSIMD in
                    normalizedPositionSIMD = positionSIMD
                }
            }
            
            Self.withSIMDAccess(to: self.unNormalizedPositionsY[bounds]) { positionSIMD in
                maxY = max(maxY, positionSIMD.max())
                minY = min(minY, positionSIMD.min())

                averageY += positionSIMD.sum()

                // Copy positions to normalized array
                Self.withSIMDAccess(to: &normalizedPositionsY[bounds]) { normalizedPositionSIMD in
                    normalizedPositionSIMD = positionSIMD
                }
            }

        }

        averageX /= Double(self.numPoints)
        averageY /= Double(self.numPoints)

        averageX = (((averageX - minX) / (maxX - minX)) * 2) - 1
        averageY = (((averageY - minX) / (maxY - minY)) * 2) - 1

        let zoomFactor = Double(self.userZoom)

        let translateX = Double((2 * self.userTranslate.x - 1))
        let translateY = Double((2 * self.userTranslate.y - 1))

        for batchIdx in 0..<self.batchCount {
            let bounds: Range<Int> = Self.batchSize*batchIdx..<Self.batchSize*(batchIdx+1)
            Self.withSIMDAccess(to: &normalizedPositionsX[bounds]) { positionsX in
                Self.withSIMDAccess(to: &normalizedPositionsY[bounds]) { positionsY in

                    positionsX = 2*((positionsX - minX) / (maxX - minX)) - 1
                    positionsY = 2*((positionsY - minY) / (maxY - minY)) - 1
                    
                    positionsX -= averageX
                    positionsY -= averageY
                    
                    positionsX *= zoomFactor
                    positionsY *= zoomFactor

                    positionsX += translateX
                    positionsY += translateY

                    positionsX = (positionsX + 1) / 2
                    positionsY = (positionsY + 1) / 2

                    positionsX.clamp(lowerBound: .zero, upperBound: .one)
                    positionsY.clamp(lowerBound: .zero, upperBound: .one)
                    
                    positionsX *= Double(self.activeBounds.width)
                    positionsY *= Double(self.activeBounds.height)

                }
            }
        }

        await MainActor.run { [normalizedPositionsX, normalizedPositionsY] in
            self.positionsX = normalizedPositionsX
            self.positionsY = normalizedPositionsY
        }
    }

    
    private func cicularize() {
        guard self.isCircularized else {
            return
        }
        
        let radialChunkSize : Double = (2*Double.pi) / Double(self.numPoints - 1)
        
        for idx in 0..<self.numPoints {
            let input = Double(idx)
            
            let newX : Double = cos(input * radialChunkSize)
            let newY : Double = idx >= (self.numPoints / 2) ? -sin(input * radialChunkSize) : sin(input * radialChunkSize)
            
            self.positionsX[idx] = newX
            self.positionsY[idx] = newY

        }
    }

}
