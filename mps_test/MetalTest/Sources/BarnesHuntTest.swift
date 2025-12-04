//
//  BarnesHuntTest.swift
//  MetalTest
//
//  Created by Owen O'Malley on 8/26/25.
//

import SwiftUI
import Observation
import Foundation

// Represents a 2D particle
struct Particle {
    var positionX: Double
    var positionY: Double
    var velocityX: Double
    var velocityY: Double
}

// Represents a square region of space
struct Region {
    var centerX: Double
    var centerY: Double
    var halfSize: Double
    
    func contains(particle: Particle) -> Bool {
        return abs(particle.positionX - centerX) <= halfSize &&
               abs(particle.positionY - centerY) <= halfSize
    }
}

// Quadtree node
class QuadTreeNode {
    var region: Region
    var particle: Particle?          // single particle if leaf
    var centerOfMassX: Double = 0.0  // average position X
    var centerOfMassY: Double = 0.0  // average position Y
    var particleCount: Int = 0
    
    var topLeft: QuadTreeNode?
    var topRight: QuadTreeNode?
    var bottomLeft: QuadTreeNode?
    var bottomRight: QuadTreeNode?
    
    init(region: Region) {
        self.region = region
    }
    
    // Insert a particle into this node
    func insert(_ newParticle: Particle) {
        // Update center of mass
        centerOfMassX = (centerOfMassX * Double(particleCount) + newParticle.positionX) / Double(particleCount + 1)
        centerOfMassY = (centerOfMassY * Double(particleCount) + newParticle.positionY) / Double(particleCount + 1)
        particleCount += 1
        
        // If no particle here yet
        if self.particle == nil && topLeft == nil {
            self.particle = newParticle
            return
        }
        
        // If this node already has one particle, subdivide
        if self.topLeft == nil {
            self.subdivide()
            if let existing = self.particle {
                self.placeIntoChild(existing)
            }
            self.particle = nil
        }
        
        placeIntoChild(newParticle)
    }
    
    private func subdivide() {
        let h = region.halfSize / 2
        topLeft = QuadTreeNode(region: Region(centerX: region.centerX - h,
                                              centerY: region.centerY + h,
                                              halfSize: h))
        topRight = QuadTreeNode(region: Region(centerX: region.centerX + h,
                                               centerY: region.centerY + h,
                                               halfSize: h))
        bottomLeft = QuadTreeNode(region: Region(centerX: region.centerX - h,
                                                 centerY: region.centerY - h,
                                                 halfSize: h))
        bottomRight = QuadTreeNode(region: Region(centerX: region.centerX + h,
                                                  centerY: region.centerY - h,
                                                  halfSize: h))
    }
    
    private func placeIntoChild(_ p: Particle) {
        if topLeft!.region.contains(particle: p) {
            topLeft!.insert(p)
        } else if topRight!.region.contains(particle: p) {
            topRight!.insert(p)
        } else if bottomLeft!.region.contains(particle: p) {
            bottomLeft!.insert(p)
        } else if bottomRight!.region.contains(particle: p) {
            bottomRight!.insert(p)
        }
    }
    
    // Compute force on a particle using Barnes-Hut approximation
    func computeForce(on p: Particle, theta: Double = 0.5) -> (fx: Double, fy: Double) {
        // If empty
        if particleCount == 0 { return (0,0) }
        
        // Distance from particle to this node's center of mass
        let dx = self.centerOfMassX - p.positionX
        let dy = self.centerOfMassY - p.positionY
        let distance = sqrt(dx*dx + dy*dy) + 1e-6 // avoid divide by zero
        
        // If this node is a single particle (leaf)
        if topLeft == nil && particle != nil {
            // Don't count force from itself
            if particle!.positionX == p.positionX && particle!.positionY == p.positionY {
                return (0,0)
            }
            let strength = 1.0 / (distance * distance)
            return (strength * dx / distance, strength * dy / distance)
        }
        
        // Size of the region
        let regionSize = region.halfSize * 2
        
        // If far enough away, approximate as one particle
        if regionSize / distance < theta {
            let strength = Double(particleCount) / (distance * distance)
            return (strength * dx / distance, strength * dy / distance)
        }
        
        // Otherwise, recurse into children
        var fx = 0.0, fy = 0.0
        if let tl = topLeft { let f = tl.computeForce(on: p, theta: theta); fx += f.fx; fy += f.fy }
        if let tr = topRight { let f = tr.computeForce(on: p, theta: theta); fx += f.fx; fy += f.fy }
        if let bl = bottomLeft { let f = bl.computeForce(on: p, theta: theta); fx += f.fx; fy += f.fy }
        if let br = bottomRight { let f = br.computeForce(on: p, theta: theta); fx += f.fx; fy += f.fy }
        return (fx, fy)
    }

}


@Observable
class NBodySimulation {
    var particles: [Particle] = []
    var simulationBounds: Region
    
    init(count: Int = 1000, bounds: Double = 1.0) {
        self.simulationBounds = Region(centerX: 0, centerY: 0, halfSize: bounds)
        self.particles = (0..<count).map { _ in
            Particle(
                positionX: Double.random(in: -bounds...bounds),
                positionY: Double.random(in: -bounds...bounds),
                velocityX: 0,
                velocityY: 0
            )
        }
    }
    
    func step(timeDelta: Double = 0.01) {
        let root = QuadTreeNode(region: simulationBounds)
        for particle in particles {
            root.insert(particle)
        }
        
        for i in 0..<particles.count {
            let force = root.computeForce(on: particles[i])
            
            // update velocity
            particles[i].velocityX += force.fx * timeDelta
            particles[i].velocityY += force.fy * timeDelta
            
            // update position
            particles[i].positionX += particles[i].velocityX * timeDelta
            particles[i].positionY += particles[i].velocityY * timeDelta
        }
    }
}

struct NBodyView: View {
    @State private var simulation = NBodySimulation(count: 1000, bounds: 1.0)
    @State private var isRunning = true
    
    // Computed property that draws quadtree nodes
    var quadnodes: some View {
        Canvas { context, size in
            guard !simulation.particles.isEmpty else { return }
            
            // Compute min/max of positions
            let minX = simulation.particles.map(\.positionX).min() ?? -1
            let maxX = simulation.particles.map(\.positionX).max() ?? 1
            let minY = simulation.particles.map(\.positionY).min() ?? -1
            let maxY = simulation.particles.map(\.positionY).max() ?? 1
            
            let rangeX = maxX - minX
            let rangeY = maxY - minY
            
            let scaleX = size.width / (rangeX == 0 ? 1 : rangeX)
            let scaleY = size.height / (rangeY == 0 ? 1 : rangeY)
            let scale = min(scaleX, scaleY) * 0.9
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            let midX = (minX + maxX) / 2
            let midY = (minY + maxY) / 2
            
            // Build tree from particles
            let root = QuadTreeNode(region: Region(centerX: 0, centerY: 0, halfSize: max(rangeX, rangeY)))
            for particle in simulation.particles {
                root.insert(particle)
            }
            
            // Recursively draw nodes
            func drawNode(_ node: QuadTreeNode?) {
                guard let node else { return }
                
                let r = node.region
                let rect = CGRect(
                    x: center.x + (r.centerX - r.halfSize - midX) * scale,
                    y: center.y - (r.centerY + r.halfSize - midY) * scale,
                    width: (r.halfSize * 2) * scale,
                    height: (r.halfSize * 2) * scale
                )
                context.stroke(Path(rect), with: .color(.green.opacity(0.3)), lineWidth: 1)
                
                drawNode(node.topLeft)
                drawNode(node.topRight)
                drawNode(node.bottomLeft)
                drawNode(node.bottomRight)
            }
            
            drawNode(root)
        }
    }
    
    var body: some View {
        ZStack {
            // Particles
            Canvas { context, size in
                guard !simulation.particles.isEmpty else { return }
                
                let minX = simulation.particles.map(\.positionX).min() ?? -1
                let maxX = simulation.particles.map(\.positionX).max() ?? 1
                let minY = simulation.particles.map(\.positionY).min() ?? -1
                let maxY = simulation.particles.map(\.positionY).max() ?? 1
                
                let rangeX = maxX - minX
                let rangeY = maxY - minY
                
                let scaleX = size.width / (rangeX == 0 ? 1 : rangeX)
                let scaleY = size.height / (rangeY == 0 ? 1 : rangeY)
                let scale = min(scaleX, scaleY) * 0.9
                let center = CGPoint(x: size.width / 2, y: size.height / 2)
                let midX = (minX + maxX) / 2
                let midY = (minY + maxY) / 2
                
                for particle in simulation.particles {
                    let x = center.x + (particle.positionX - midX) * scale
                    let y = center.y - (particle.positionY - midY) * scale
                    let rect = CGRect(x: x, y: y, width: 3, height: 3)
                    context.fill(Path(ellipseIn: rect), with: .color(.blue))
                }
            }
            
            // Overlay quadtree visualization
            quadnodes
        }
        .background(Color.black)
        .onAppear {
            Task {
                while isRunning {
                    simulation.step()
                    try? await Task.sleep(for: .milliseconds(16)) // ~60 FPS
                }
            }
        }
    }
}
