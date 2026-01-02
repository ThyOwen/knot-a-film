//
//  BufferGroup.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 1/2/26.
//

import SharedWithMetal
import Metal

protocol MTLBufferGroup {
    static func makeBuffers(size: UInt32, device: MTLDevice, options: MTLResourceOptions) -> Self
}

public struct NodeBufferGroup : MTLBufferGroup {
    public var topLeftBuffer: MTLBuffer
    public var bottomRightBuffer: MTLBuffer
    public var centerOfMassBuffer: MTLBuffer
    public var totalMassBuffer: MTLBuffer
    public var startBuffer: MTLBuffer
    public var endBuffer: MTLBuffer
    public var isLeafBuffer: MTLBuffer
    
    static func makeBuffers(size: UInt32, device: MTLDevice, options: MTLResourceOptions) -> Self {
        let topLeftBuffer = device.makeBuffer(length: MemoryLayout<NodeMemberDataFloat2>.stride, options: options)!
        let bottomRightBuffer = device.makeBuffer(length: MemoryLayout<NodeMemberDataFloat2>.stride, options: options)!
        let centerOfMassBuffer = device.makeBuffer(length: MemoryLayout<NodeMemberDataFloat2>.stride, options: options)!
        let totalMassBuffer = device.makeBuffer(length: MemoryLayout<NodeMemberDataFloat>.stride, options: options)!
        let startBuffer = device.makeBuffer(length: MemoryLayout<NodeMemberDataUInt32>.stride, options: options)!
        let endBuffer = device.makeBuffer(length: MemoryLayout<NodeMemberDataUInt32>.stride, options: options)!
        let isLeafBuffer = device.makeBuffer(length: MemoryLayout<NodeMemberDataBool>.stride, options: options)!
        
        let numNodesPtr = topLeftBuffer.contents().bindMemory(to: NodeMemberDataFloat2.self, capacity: 1)
        numNodesPtr.pointee.size = size

        let numNodesBottomPtr = bottomRightBuffer.contents().bindMemory(to: NodeMemberDataFloat2.self, capacity: 1)
        numNodesBottomPtr.pointee.size = size

        let numCenterPtr = centerOfMassBuffer.contents().bindMemory(to: NodeMemberDataFloat2.self, capacity: 1)
        numCenterPtr.pointee.size = size

        let numTotalMassPtr = totalMassBuffer.contents().bindMemory(to: NodeMemberDataFloat.self, capacity: 1)
        numTotalMassPtr.pointee.size = size

        let numStartPtr = startBuffer.contents().bindMemory(to: NodeMemberDataUInt32.self, capacity: 1)
        numStartPtr.pointee.size = size

        let numEndPtr = endBuffer.contents().bindMemory(to: NodeMemberDataUInt32.self, capacity: 1)
        numEndPtr.pointee.size = size

        let numIsLeafPtr = isLeafBuffer.contents().bindMemory(to: NodeMemberDataBool.self, capacity: 1)
        numIsLeafPtr.pointee.size = size
        
        return .init(topLeftBuffer: consume topLeftBuffer,
                     bottomRightBuffer: consume bottomRightBuffer,
                     centerOfMassBuffer: consume centerOfMassBuffer,
                     totalMassBuffer: consume totalMassBuffer,
                     startBuffer: consume startBuffer,
                     endBuffer: consume endBuffer,
                     isLeafBuffer: consume isLeafBuffer)
    }
}

public struct BodyBufferGroup : MTLBufferGroup {
    public var massBuffer: MTLBuffer
    public var radiusBuffer: MTLBuffer
    public var positionBuffer: MTLBuffer
    public var velocityBuffer: MTLBuffer
    public var accelerationBuffer: MTLBuffer
    public var initialIdxBuffer: MTLBuffer
    public var offsetsBuffer: MTLBuffer
    
    public var massAltBuffer: MTLBuffer
    public var positionAltBuffer: MTLBuffer
    public var initialIdxAltBuffer: MTLBuffer
    
    static func makeBuffers(size: UInt32, device: MTLDevice, options: MTLResourceOptions) -> Self {
        let massBuffer = device.makeBuffer(length: MemoryLayout<BodyMemberDataFloat>.stride, options: .storageModeShared)!
        let radiusBuffer = device.makeBuffer(length: MemoryLayout<BodyMemberDataFloat>.stride, options: .storageModeShared)!
        let positionBuffer = device.makeBuffer(length: MemoryLayout<BodyMemberDataFloat2>.stride, options: .storageModeShared)!
        let velocityBuffer = device.makeBuffer(length: MemoryLayout<BodyMemberDataFloat2>.stride, options: .storageModeShared)!
        let accelerationBuffer = device.makeBuffer(length: MemoryLayout<BodyMemberDataFloat2>.stride, options: .storageModeShared)!
        let initialIdxBuffer = device.makeBuffer(length: MemoryLayout<BodyMemberDataUInt32>.stride, options: .storageModeShared)!
        let offsetsBuffer = device.makeBuffer(length: MemoryLayout<BodyMemberDataUInt32>.stride, options: .storageModeShared)!
        
        let massAltBuffer = device.makeBuffer(length: MemoryLayout<BodyMemberDataFloat>.stride, options: .storageModeShared)!
        let positionAltBuffer = device.makeBuffer(length: MemoryLayout<BodyMemberDataFloat2>.stride, options: .storageModeShared)!
        let initialIdxAltBuffer = device.makeBuffer(length: MemoryLayout<BodyMemberDataUInt32>.stride, options: .storageModeShared)!
        
        let numInstancesPtr = massBuffer.contents().bindMemory(to: BodyMemberDataFloat.self, capacity: 1)
        numInstancesPtr.pointee.size = size

        let numRadiusPtr = radiusBuffer.contents().bindMemory(to: BodyMemberDataFloat.self, capacity: 1)
        numRadiusPtr.pointee.size = size

        let numPositionPtr = positionBuffer.contents().bindMemory(to: BodyMemberDataFloat2.self, capacity: 1)
        numPositionPtr.pointee.size = size

        let numVelocityPtr = velocityBuffer.contents().bindMemory(to: BodyMemberDataFloat2.self, capacity: 1)
        numVelocityPtr.pointee.size = size

        let numAccelerationPtr = accelerationBuffer.contents().bindMemory(to: BodyMemberDataFloat2.self, capacity: 1)
        numAccelerationPtr.pointee.size = size

        let numInitialIdxPtr = initialIdxBuffer.contents().bindMemory(to: BodyMemberDataUInt32.self, capacity: 1)
        numInitialIdxPtr.pointee.size = size
        
        let numOffsetsPtr = offsetsBuffer.contents().bindMemory(to: BodyMemberDataUInt32.self, capacity: 1)
        numOffsetsPtr.pointee.size = size
        
        let massAltPtr = massAltBuffer.contents().bindMemory(to: BodyMemberDataFloat.self, capacity: 1)
        massAltPtr.pointee.size = size

        let positionAltPtr = positionAltBuffer.contents().bindMemory(to: BodyMemberDataFloat2.self, capacity: 1)
        positionAltPtr.pointee.size = size

        let initialIdxAltPtr = initialIdxAltBuffer.contents().bindMemory(to: BodyMemberDataUInt32.self, capacity: 1)
        initialIdxAltPtr.pointee.size = size
        
        return BodyBufferGroup(massBuffer: consume massBuffer,
                               radiusBuffer: consume radiusBuffer,
                               positionBuffer: consume positionBuffer,
                               velocityBuffer: consume velocityBuffer,
                               accelerationBuffer: consume accelerationBuffer,
                               initialIdxBuffer: consume initialIdxBuffer,
                               offsetsBuffer: consume offsetsBuffer,
                               massAltBuffer: consume massAltBuffer,
                               positionAltBuffer: consume positionAltBuffer,
                               initialIdxAltBuffer: consume initialIdxAltBuffer)
    }
    
    public func setSentinal(_ numBodies : UInt32, _ numEdges : UInt32) {
        // Write sentinel at offsets[numBodies] = total number of flattened edges so mesh shaders
        // can safely read offsets[gid+1].
        let numOffsetsPtr = offsetsBuffer.contents().assumingMemoryBound(to: UInt32.self)
        numOffsetsPtr[Int(numBodies + 1)] = numEdges
    }
}

public struct EdgesBufferGroup : MTLBufferGroup {
    public var edgeTerminationsBuffer : MTLBuffer
    public var edgeTerminationsSortedBuffer : MTLBuffer
    public var edgeAnglesBuffer : MTLBuffer
    public var edgeSourcesBuffer : MTLBuffer
    
    static func makeBuffers(size: UInt32, device: MTLDevice, options: MTLResourceOptions) -> Self {
        let edgeTerminationsBuffer = device.makeBuffer(length: MemoryLayout<EdgeMemberDataUInt32>.stride, options: .storageModeShared)!
        let edgeSourcesBuffer = device.makeBuffer(length: MemoryLayout<EdgeMemberDataUInt32>.stride, options: .storageModeShared)!
        let edgeTerminationsSortedBuffer = device.makeBuffer(length: MemoryLayout<EdgeMemberDataUInt32>.stride, options: .storageModeShared)!
        let edgeAnglesBuffer = device.makeBuffer(length: MemoryLayout<EdgeMemberDataUInt32>.stride, options: .storageModeShared)!
        
        let edgeSourcesPointer = edgeSourcesBuffer.contents().bindMemory(to: EdgeMemberDataUInt32.self, capacity: 1)
        edgeSourcesPointer.pointee.size = size
        
        let edgeTerminationsPointer = edgeTerminationsBuffer.contents().bindMemory(to: EdgeMemberDataUInt32.self, capacity: 1)
        edgeTerminationsPointer.pointee.size = size
        
        let sortedEdgeIndiciesPointer = edgeTerminationsSortedBuffer.contents().bindMemory(to: EdgeMemberDataUInt32.self, capacity: 1)
        sortedEdgeIndiciesPointer.pointee.size = size
        
        let edgeAnglesPointer = edgeAnglesBuffer.contents().bindMemory(to: EdgeMemberDataUInt32.self, capacity: 1)
        edgeAnglesPointer.pointee.size = size
        
        return EdgesBufferGroup(edgeTerminationsBuffer: consume edgeTerminationsBuffer,
                                edgeTerminationsSortedBuffer: consume edgeTerminationsSortedBuffer,
                                edgeAnglesBuffer: consume edgeAnglesBuffer,
                                edgeSourcesBuffer: consume edgeSourcesBuffer)
    }

}
