//
//  GraphOld.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 1/30/25.
//

import Foundation
import Observation
import SwiftUICore

import Accelerate
/*
@Observable public final class Graph {
    
    public typealias N = Double
    public typealias SIMDType = SIMD64<N>
    public typealias SubSIMDType = SIMD32<N>

    private let batchCount : Int
    private let numPoints : Int
    private let batchSize : Int
    
    public var isCircularized : Bool = false

    public let nodes : [Movie]
    public let edges : [MovieEdge<N>]
    public private(set) var positionsX : ContiguousArray<N>
    public private(set) var positionsY : ContiguousArray<N>
    @ObservationIgnored public private(set) var unNormalizedPositionsX : ManagedBuffer<Int, N>
    @ObservationIgnored public private(set) var unNormalizedPositionsY : ManagedBuffer<Int, N>
    @ObservationIgnored private var velocitiesX : ManagedBuffer<Int, N>
    @ObservationIgnored private var velocitiesY : ManagedBuffer<Int, N>

    public var activeBounds : CGSize
    public static let passiveBound : N = 100
    
    public var userTranslate : UnitPoint = .center
    public var userZoom : CGFloat = 1
    public var userZoomCenter : UnitPoint = .zero
    
    private let repulsionStrength: N = 1000
    private let attractionStrength: N = 0.001
    private let damping: N = 0.85
    
    public private(set) var simulationTask : Task<Void, Never>? = nil
    public private(set) var visualizeTask : Task<Void, Never>? = nil
    
    public init(of watchedMovies : consuming [Movie], batchSize : Int = 1024) {
        
        let numOfPoints : Int = watchedMovies.count

        let array = ContiguousArray<N>.init(unsafeUninitializedCapacity: numOfPoints) { buffer, initializedCount in
            for idx in 0..<numOfPoints {
                let value = N.random(in: -Self.passiveBound...Self.passiveBound)
                buffer[idx] = value
            }
            initializedCount = numOfPoints  // Set final count here instead of incrementing
        }

        self.positionsX = array
        self.positionsY = array
        self.unNormalizedPositionsX = array
        self.unNormalizedPositionsY = consume array
        self.velocitiesX = .init(repeating: 0.0, count: numOfPoints)
        self.velocitiesY = .init(repeating: 0.0, count: numOfPoints)

        self.numPoints = numOfPoints
        self.batchSize = batchSize
        self.batchCount = numOfPoints / batchSize
        
        self.edges = Self.findConnections(between: watchedMovies)

        self.activeBounds = .init(width: CGFloat(Self.passiveBound), height: CGFloat(Self.passiveBound))
        
        
        self.nodes = consume watchedMovies
    }
    
    //MARK: - Connections
    private static func findConnections(between watchedMovies : borrowing [Movie]) -> [MovieEdge<N>] {
        
        var edges : [MovieEdge<N>] = []
        
        for (aIdx, movieA) in watchedMovies.enumerated() {
            for movieB in watchedMovies[(aIdx + 1)...] { // Only consider movies after indexA
                if let edge = try? MovieEdge<N>(movieA, movieB) {
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
                //await self.visualizeTick()
                try? await Task.sleep(for: .seconds(0.01))
            }
        }
        
        if !self.isCircularized {
            self.simulationTask = Task.detached(priority: .high) {
                for _ in 0...100 {
                    let start = Date.now
                    await self.simulateTick()
                    print(Date.now.timeIntervalSince1970 - start.timeIntervalSince1970)
                    try? await Task.sleep(for: .seconds(0.02))
                }
                
                print("done")
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
        
        for aBatchIdx in 0..<self.batchCount {
            for bBatchIdx in (aBatchIdx + 1)..<self.batchCount {
                
                let aRange = Self.batchSize*aBatchIdx..<Self.batchSize*(aBatchIdx+1)
                let bRange = Self.batchSize*bBatchIdx..<Self.batchSize*(bBatchIdx+1)
                
                Self.withRangedAccess(to: newPositionsX, aRange: aRange, bRange: bRange) { aNodePositionX, bNodePositionX in
                    Self.withRangedAccess(to: newPositionsY, aRange: aRange, bRange: bRange) { aNodePositionY, bNodePositionY in
                        
                        var deltaX : Array<N> = .init(unsafeUninitializedCapacity: self.numPoints) { buffer, initializedCount in
                            vDSP_vsub(bNodePositionX, 1,
                                      aNodePositionX, 1,
                                      buffer.baseAddress!, 1,
                                      vDSP_Length(self.numPoints))
                        }
                        
                        var deltaY : Array<N> = .init(unsafeUninitializedCapacity: self.numPoints) { buffer, initializedCount in
                            vDSP_vsub(bNodePositionY, 1,
                                      aNodePositionY, 1,
                                      buffer.baseAddress!, 1,
                                      vDSP_Length(self.numPoints))
                        }
                        
                        let distance = vDSP.hypot(deltaX, deltaY)
                        //let distanceMask = distance .< 1.0
                        //distance.replace(with: 1.0, where: distanceMask)
                        
                        vDSP.divide(deltaX, distance, result: &deltaX)
                        vDSP.divide(deltaX, distance, result: &deltaY)
                        
                        vDSP.square(distance, result: &distance)
                        
                        let force = vDSP.divide(self.repulsionStrength, consume distance)
                        
                        let forceX = vDSP.multiply(force, consume deltaX)
                        let forceY = vDSP.multiply(consume force, consume deltaY)
                        
                        Self.withRangedMutableAccess(to: newVelocitiesX, aRange: aRange, bRange: bRange) { aNodeVelocityX, bNodeVelocityX in
                            Self.withRangedMutableAccess(to: newVelocitiesY, aRange: aRange, bRange: bRange) { aNodeVelocityY, bNodeVelocityY in
                            
                                forceX.withContiguousStorageIfAvailable { forcePointerX in
                                    vDSP_vsub(aNodeVelocityX, 1,
                                              forcePointerX, 1,
                                              aNodeVelocityX, 1,
                                              vDSP_Length(self.numPoints))
                                    vDSP_vadd(bNodeVelocityX, 1,
                                              forcePointerX, 1,
                                              bNodeVelocityX, 1,
                                              vDSP_Length(self.numPoints))
                                }
                                
                                forceY.withContiguousStorageIfAvailable { forcePointerY in
                                    vDSP_vsub(aNodeVelocityY, 1,
                                              forcePointerY, 1,
                                              aNodeVelocityY, 1,
                                              vDSP_Length(self.numPoints))
                                    vDSP_vadd(bNodeVelocityY, 1,
                                              forcePointerY, 1,
                                              bNodeVelocityY, 1,
                                              vDSP_Length(self.numPoints))
                                }
                            }
                        }
                    }
                }
                
            }
        }

        // Apply attraction along edges
        await withTaskGroup(of: <#T##Sendable.Type#>) { group in
            for edge in self.edges {
                group.addTask {
                    
                }
            }
        }

        
        // Update positions and apply damping
        for batchIdx in 0..<self.batchCount {
            let bounds: Range<Int> = Self.batchSize*batchIdx..<Self.batchSize*(batchIdx+1)
            
            newPositionsX[bounds].withUnsafeMutablePointerToElements { positionsPointerX in
                newPositionsY[bounds].withUnsafeMutablePointerToElements { positionsPointerY in
                    newVelocitiesX[bounds].withUnsafeMutablePointerToElements { velocitiesPointerX in
                        newVelocitiesY[bounds].withUnsafeMutablePointerToElements { velocitiesPointerY in
                            
                            withUnsafePointer(to: self.damping) { dampingPointer in
                                vDSP_vsmul(velocitiesPointerX, 1,
                                           dampingPointer,
                                           velocitiesPointerX, 1,
                                           vDSP_Length(self.numPoints))
                                
                                vDSP_vsmul(velocitiesPointerY, 1,
                                           dampingPointer,
                                           velocitiesPointerY, 1,
                                           vDSP_Length(self.numPoints))
                            }

                            vDSP_vadd(positionsPointerX, 1,
                                      velocitiesPointerX, 1,
                                      positionsPointerX, 1,
                                      vDSP_Length(self.numPoints))
                            
                            vDSP_vadd(positionsPointerY, 1,
                                      velocitiesPointerY, 1,
                                      positionsPointerY, 1,
                                      vDSP_Length(self.numPoints))

                        }
                    }
                }
            }
        }

        await MainActor.run { [newPositions] in
            self.unNormalizedPositionsX = consume newPositionsX
            self.unNormalizedPositionsY = consume newPositionsY
        }
        self.velocitiesX = consume newVelocities
    }
    /*
    private func visualizeTick() async {
        var normalizedPositions = ContiguousArray<N>.init(repeating: 0.0, count: self.positions.count)

        var maxX: N = 0
        var maxY: N = 0
        var minX: N = 0
        var minY: N = 0

        var averageX: N = 0
        var averageY: N = 0

        for batchIdx in 0..<self.batchCount {
            let bounds: Range<Int> = (Self.batchSize * batchIdx)..<(Self.batchSize * (batchIdx + 1))

            Self.withSIMDAccess(to: self.unNormalizedPositions[bounds]) { positionSIMD in
                maxX = max(maxX, positionSIMD.evenHalf.max())
                maxY = max(maxY, positionSIMD.oddHalf.max())

                minX = min(minX, positionSIMD.evenHalf.min())
                minY = min(minY, positionSIMD.oddHalf.min())

                averageX += positionSIMD.evenHalf.sum()
                averageY += positionSIMD.oddHalf.sum()

                // Copy positions to normalized array
                Self.withSIMDAccess(to: &normalizedPositions[bounds]) { normalizedPositionSIMD in
                    normalizedPositionSIMD = positionSIMD
                }
            }
        }

        averageX /= N(self.numPoints)
        averageY /= N(self.numPoints)

        averageX = (((averageX - minX) / (maxX - minX)) * 2) - 1
        averageY = (((averageY - minX) / (maxY - minY)) * 2) - 1

        let zoomFactor = N(self.userZoom)

        let translateX = N((2 * self.userTranslate.x - 1))
        let translateY = N((2 * self.userTranslate.y - 1))

        for batchIdx in 0..<self.batchCount {
            let bounds: Range<Int> = Self.batchSize*batchIdx..<Self.batchSize*(batchIdx+1)
            Self.withSIMDAccess(to: &normalizedPositions[bounds]) { positions in

                positions.evenHalf = 2*((positions.evenHalf - minX) / (maxX - minX)) - 1
                positions.oddHalf = 2*((positions.oddHalf - minY) / (maxY - minY)) - 1
                
                positions.evenHalf += averageX
                positions.oddHalf += averageY
                
                positions *= zoomFactor

                positions.evenHalf += translateX
                positions.oddHalf += translateY

                positions = (positions + 1) / 2
                
                positions.clamp(lowerBound: .zero, upperBound: .one)
                
                // Scale to screen space
                positions.evenHalf *= N(self.activeBounds.width)
                positions.oddHalf *= N(self.activeBounds.height)
            }
        }

        await MainActor.run { [normalizedPositions] in
            self.positions = consume normalizedPositions
        }
    }
     */
    private func cicularize() {
        guard self.isCircularized else {
            return
        }
        
        let moviesCount = self.positions.count / 2
        
        let radialChunkSize : N = (2*N.pi) / N(moviesCount - 1)
        
        for idx in stride(from: 0, to: self.positions.count, by: 2) {
            let input = N(idx)
            
            let x : N = cos(input * radialChunkSize)
            let y : N = sin(input * radialChunkSize)
            
            self.positions[idx] = x
            self.positions[idx + 1] = idx >= moviesCount ? -y : y
        }
    }

}
*/
