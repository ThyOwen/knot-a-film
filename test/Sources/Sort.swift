//
//  Sort.swift
//  MetalTest
//
//  Created by Owen O'Malley on 12/28/25.
//

import Foundation
import Metal

struct MetalTests {
    public static func sumTest() {
        var N : UInt32 = 1024
        
        let device = MTLCreateSystemDefaultDevice()!
        let library = try! device.makeDefaultLibrary(bundle: .module)

        let prefixSumFunction = library.makeFunction(name: "prefixSum")!
        let prefixSumPipeline = try! device.makeComputePipelineState(function: prefixSumFunction)
        
        let dataSize = MemoryLayout<UInt32>.stride * Int(N)
        let data = (0..<N).map { UInt32($0) }
        let inDataBuffer = device.makeBuffer(bytes: consume data, length: dataSize, options: .storageModeShared)!
        let outDataBuffer = device.makeBuffer(length: dataSize, options: .storageModeShared)!
        
        let commandBuffer = device.makeCommandQueue()!.makeCommandBuffer()!
        
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        
        encoder.setComputePipelineState(prefixSumPipeline)
        
        encoder.setBuffer(inDataBuffer, offset: 0, index: 0)
        encoder.setBuffer(outDataBuffer, offset: 0, index: 1)
        encoder.setBytes(&N, length: MemoryLayout<UInt32>.size, index: 2)
        
        let gridSize = MTLSize(width: 1, height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let outDataBufferPtr = outDataBuffer.contents().assumingMemoryBound(to: UInt32.self)
        
        for i in 0..<N {
            print("\(i): \(outDataBufferPtr[Int(i)])")
        }


    }
    
    public static func sumGlobalTest() {
        var N : UInt32 = 1024
        
        let device = MTLCreateSystemDefaultDevice()!
        let library = try! device.makeDefaultLibrary(bundle: .module)

        let prefixSumFunction = library.makeFunction(name: "prefixGlobalSum")!
        let prefixSumPipeline = try! device.makeComputePipelineState(function: prefixSumFunction)
        
        let dataSize = MemoryLayout<UInt32>.stride * Int(N)
        let data = (0..<N).map { UInt32($0) }
        let inDataBuffer = device.makeBuffer(bytes: consume data, length: dataSize, options: .storageModeShared)!
        let outDataBuffer = device.makeBuffer(length: dataSize, options: .storageModeShared)!
        
        let commandBuffer = device.makeCommandQueue()!.makeCommandBuffer()!
        
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        
        encoder.setComputePipelineState(prefixSumPipeline)
        
        encoder.setBuffer(inDataBuffer, offset: 0, index: 0)
        encoder.setBuffer(outDataBuffer, offset: 0, index: 1)
        encoder.setBytes(&N, length: MemoryLayout<UInt32>.size, index: 2)

        
        let threadsPerThreadgroup = 32
        let threadgroupCount = (Int(N) + threadsPerThreadgroup - 1) / threadsPerThreadgroup

        let tgSize = MTLSize(width: threadsPerThreadgroup, height: 1, depth: 1)
        let gridSize = MTLSize(width: threadgroupCount, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: tgSize)
        
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let outDataBufferPtr = outDataBuffer.contents().assumingMemoryBound(to: UInt32.self)
        
        for i in 0..<N {
            print("\(i): \(outDataBufferPtr[Int(i)])")
        }


    }
    
    public static func sortTest() {
        var N: UInt32 = 64
        
        let device = MTLCreateSystemDefaultDevice()!
        let library = try! device.makeDefaultLibrary(bundle: .module)

        let sortFunction = library.makeFunction(name: "radixSort")!
        let pipelineState = try! device.makeComputePipelineState(function: sortFunction)
        
        let dataSize = MemoryLayout<UInt32>.stride * Int(N)
        var data = (0..<N).map { $0 } // _ in UInt32.random(in: 0..<100000) }
        
        let inDataBuffer = device.makeBuffer(bytes: &data, length: dataSize, options: .storageModeShared)!
        let outDataBuffer = device.makeBuffer(length: dataSize, options: .storageModeShared)!
        
        let commandQueue = device.makeCommandQueue()!
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        
        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(inDataBuffer, offset: 0, index: 0)
        encoder.setBuffer(outDataBuffer, offset: 0, index: 1)
        encoder.setBytes(&N, length: MemoryLayout<UInt32>.size, index: 2)
        
        let threadsPerThreadgroup = MTLSize(width: 32, height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(width: 1, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // verification
        let gpuVals = outDataBuffer.contents().bindMemory(to: UInt32.self, capacity: Int(N))
        
        let sortedGpuData = Array(UnsafeBufferPointer(start: gpuVals, count: Int(N)))

        let expectedData = data.sorted { (a, b) in
            return a < b
        }
                
        var isMatch = true
        
        for i in 0..<Int(N) {
            if sortedGpuData[i] != expectedData[i] {
                isMatch = false
            }
        }
        
        print("--- UInt32 Sort Results ---")
        print("First 5 (GPU):", sortedGpuData.prefix(10))
        print("Matches CPU: \(isMatch)")
    }
    
    public static func sortKVTest() {
        var N: UInt32 = 1024
        
        let device = MTLCreateSystemDefaultDevice()!
        let library = try! device.makeDefaultLibrary(bundle: .module)

        let sortFunction = library.makeFunction(name: "radixSortKV")!
        let pipelineState = try! device.makeComputePipelineState(function: sortFunction)
        
        var keys = (0..<N).map { _ in Float.random(in: -5000...5000) }
        var values = (0..<N).map { $0 } // _ in UInt32.random(in: 0...100_000) }
        
        let keySize = MemoryLayout<Float>.stride * Int(N)
        let valSize = MemoryLayout<UInt32>.stride * Int(N)
        
        let keysInBuf = device.makeBuffer(bytes: &keys, length: keySize, options: .storageModeShared)!
        let valsInBuf = device.makeBuffer(bytes: &values, length: valSize, options: .storageModeShared)!
        let keysOutBuf = device.makeBuffer(length: keySize, options: .storageModeShared)!
        let valsOutBuf = device.makeBuffer(length: valSize, options: .storageModeShared)!
        
        let commandQueue = device.makeCommandQueue()!
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        
        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(keysInBuf, offset: 0, index: 0)
        encoder.setBuffer(valsInBuf, offset: 0, index: 1)
        encoder.setBuffer(keysOutBuf, offset: 0, index: 2)
        encoder.setBuffer(valsOutBuf, offset: 0, index: 3)
        encoder.setBytes(&N, length: MemoryLayout<UInt32>.size, index: 4)
        
        let threadsPerThreadgroup = MTLSize(width: 32, height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(width: 1, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // verification
        let gpuKeys = keysOutBuf.contents().bindMemory(to: Float.self, capacity: Int(N))
        let gpuVals = valsOutBuf.contents().bindMemory(to: UInt32.self, capacity: Int(N))
        
        let sortedGpuKeys = Array(UnsafeBufferPointer(start: gpuKeys, count: Int(N)))
        let sortedGpuVals = Array(UnsafeBufferPointer(start: gpuVals, count: Int(N)))

        let cpuSorted = zip(keys, values).enumerated().sorted { (a, b) in
            if a.element.0 != b.element.0 {
                return a.element.0 < b.element.0
            }
            return a.offset < b.offset
        }
        
        let expectedKeys = cpuSorted.map { $0.element.0 }
        let expectedVals = cpuSorted.map { $0.element.1 }
        
        var isMatch = true
        
        for i in 0..<Int(N) {
            if sortedGpuKeys[i] != expectedKeys[i] || sortedGpuVals[i] != expectedVals[i] {
                isMatch = false
            }
        }
        
        print("--- Float Key-Value Sort Results ---")
        print("First 5 Keys (GPU):", sortedGpuKeys.prefix(10))
        print("First 5 Vals (GPU):", sortedGpuVals.prefix(10))
        print("Matches CPU: \(isMatch)")
    }
    
    public static func sortKV2Test() {
        var N: UInt32 = 1024
        
        let device = MTLCreateSystemDefaultDevice()!
        let library = try! device.makeDefaultLibrary(bundle: .module)

        let sortFunction = library.makeFunction(name: "radixSortKV2")!
        let pipelineState = try! device.makeComputePipelineState(function: sortFunction)
        
        var keys = (0..<N).map { _ in Float.random(in: -5000...5000) }
        var values = (0..<N).map { $0 } // _ in UInt32.random(in: 0...100_000) }
        
        let keySize = MemoryLayout<Float>.stride * Int(N)
        let valSize = MemoryLayout<UInt32>.stride * Int(N)
        
        let keysInBuf = device.makeBuffer(bytes: &keys, length: keySize, options: .storageModeShared)!
        let valsInBuf = device.makeBuffer(bytes: &values, length: valSize, options: .storageModeShared)!
        let keysOutBuf = device.makeBuffer(length: keySize, options: .storageModeShared)!
        let valsOutBuf = device.makeBuffer(length: valSize, options: .storageModeShared)!
        
        let commandQueue = device.makeCommandQueue()!
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        
        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(keysInBuf, offset: 0, index: 0)
        encoder.setBuffer(valsInBuf, offset: 0, index: 1)
        encoder.setBuffer(keysOutBuf, offset: 0, index: 2)
        encoder.setBuffer(valsOutBuf, offset: 0, index: 3)
        encoder.setBytes(&N, length: MemoryLayout<UInt32>.size, index: 4)
        
        let threadsPerThreadgroup = MTLSize(width: 32, height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(width: 1, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // verification
        let gpuKeys = keysOutBuf.contents().bindMemory(to: Float.self, capacity: Int(N))
        let gpuVals = valsOutBuf.contents().bindMemory(to: UInt32.self, capacity: Int(N))
        
        let sortedGpuKeys = Array(UnsafeBufferPointer(start: gpuKeys, count: Int(N)))
        let sortedGpuVals = Array(UnsafeBufferPointer(start: gpuVals, count: Int(N)))

        let cpuSorted = zip(keys, values).enumerated().sorted { (a, b) in
            if a.element.0 != b.element.0 {
                return a.element.0 < b.element.0
            }
            return a.offset < b.offset
        }
        
        let expectedKeys = cpuSorted.map { $0.element.0 }
        let expectedVals = cpuSorted.map { $0.element.1 }
        
        var isMatch = true
        
        for i in 0..<Int(N) {
            if sortedGpuKeys[i] != expectedKeys[i] || sortedGpuVals[i] != expectedVals[i] {
                isMatch = false
            }
        }
        
        print("--- Float Key-Value Sort Results ---")
        print("First 5 Keys (GPU):", sortedGpuKeys.prefix(10))
        print("First 5 Vals (GPU):", sortedGpuVals.prefix(10))
        print("Matches CPU: \(isMatch)")
    }
}
