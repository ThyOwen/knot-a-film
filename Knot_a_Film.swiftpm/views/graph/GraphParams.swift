//
//  GraphParams.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 12/28/25.
//

import SharedWithMetal
import Foundation
import class Metal.MTLFunctionConstantValues

extension GraphParams {

    public static func makeDefault( numBodies: UInt32, numConnections: UInt32 ) -> GraphParams {
        
        var p = GraphParams()
        
        p.numBodies = numBodies
        p.numConnections = numConnections
        
        p.maxDepth = 8
        p.numNodes = (UInt32(pow(4.0, Double(p.maxDepth + 1))) - 1) / 3
        p.leafLimit = (UInt32(pow(4.0, Double(p.maxDepth))) - 1) / 3
        
        p.blockSize = 32
        p.useBarnes = true
        p.computeConnections = true
        
        p.physics.springConstant = 1.0
        p.physics.edgeRepulsion = 1.0
        p.physics.edgeAttraction = 1.0
        p.physics.epsilon = 0.01
        p.physics.dt = 0.01
        p.physics.theta = 0.1
        p.physics.collisionThreshold = 0.001
        p.physics.damping = 0.95
        
        return p
    }
    
    public mutating func addMetalFunctionConstants( to metalFunctionConstants: borrowing MTLFunctionConstantValues ) {
        
        metalFunctionConstants.setConstantValue(
            &self.numBodies,
            type: .uint,
            index: Int(FC_NUM_BODIES)
        )
        
        metalFunctionConstants.setConstantValue(
            &self.numConnections,
            type: .uint,
            index: Int(FC_NUM_CONNECTIONS)
        )
        
        metalFunctionConstants.setConstantValue(
            &self.numNodes,
            type: .uint,
            index: Int(FC_NUM_NODES)
        )
        
        metalFunctionConstants.setConstantValue(
            &self.leafLimit,
            type: .uint,
            index: Int(FC_LEAF_LIMIT)
        )
        
        metalFunctionConstants.setConstantValue(
            &self.blockSize,
            type: .int,
            index: Int(FC_BLOCK_SIZE)
        )
        
        metalFunctionConstants.setConstantValue(
            &self.useBarnes,
            type: .bool,
            index: Int(FC_USE_BARNES)
        )
        
        metalFunctionConstants.setConstantValue(
            &self.computeConnections,
            type: .bool,
            index: Int(FC_COMPUTE_CONNECTIONS)
        )
        
        metalFunctionConstants.setConstantValue(
            &self.physics.springConstant,
            type: .float,
            index: Int(FC_SPRING_CONSTANT)
        )
        
        metalFunctionConstants.setConstantValue(
            &self.physics.edgeRepulsion,
            type: .float,
            index: Int(FC_EDGE_REPULSION)
        )
        
        metalFunctionConstants.setConstantValue(
            &self.physics.edgeAttraction,
            type: .float,
            index: Int(FC_EDGE_ATTRACTION)
        )
        
        metalFunctionConstants.setConstantValue(
            &self.physics.epsilon,
            type: .float,
            index: Int(FC_EPSILON)
        )
        
        metalFunctionConstants.setConstantValue(
            &self.physics.dt,
            type: .float,
            index: Int(FC_DT)
        )
        
        metalFunctionConstants.setConstantValue(
            &self.physics.theta,
            type: .float,
            index: Int(FC_THETA)
        )
        
        metalFunctionConstants.setConstantValue(
            &self.physics.collisionThreshold,
            type: .float,
            index: Int(FC_COLLISION_THRESHOLD)
        )
        
        metalFunctionConstants.setConstantValue(
            &self.physics.damping,
            type: .float,
            index: Int(FC_DAMPING)
        )
    }
    
}
