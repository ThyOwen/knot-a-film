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
        var N : Int32 = 1024
        
        let device = MTLCreateSystemDefaultDevice()!
        let library = try! device.makeDefaultLibrary(bundle: .module)

        let prefixSumFunction = library.makeFunction(name: "prefixSum")!
        let prefixSumPipeline = try! device.makeComputePipelineState(function: prefixSumFunction)
        
        let dataSize = MemoryLayout<Int32>.stride * Int(N)
        let data = (0..<N).map { Int32($0) }
        let inDataBuffer = device.makeBuffer(bytes: consume data, length: dataSize, options: .storageModeShared)!
        let outDataBuffer = device.makeBuffer(length: dataSize, options: .storageModeShared)!
        
        let commandBuffer = device.makeCommandQueue()!.makeCommandBuffer()!
        
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        
        encoder.setComputePipelineState(prefixSumPipeline)
        
        encoder.setBytes(&N, length: MemoryLayout<Int32>.size, index: 0)
        encoder.setBuffer(inDataBuffer, offset: 0, index: 1)
        encoder.setBuffer(outDataBuffer, offset: 0, index: 2)
        
        let gridSize = MTLSize(width: 1, height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let outDataBufferPtr = outDataBuffer.contents().assumingMemoryBound(to: Int32.self)
        
        for i in 0..<N {
            print("\(i): \(outDataBufferPtr[Int(i)])")
        }


    }
    
    public static func swiftSumTest() {
        func blellochScan(_ a: inout [Int]) {
            let n = a.count
            // Up-sweep (reduce)
            var offset = 1
            var d = n >> 1
            while d > 0 {
                for i in 0..<d {
                    let ai = offset * (2*i + 1) - 1
                    let bi = offset * (2*i + 2) - 1
                    a[bi] += a[ai]
                }
                offset *= 2
                d >>= 1
            }

            // Down-sweep
            a[n - 1] = 0
            d = 1
            offset >>= 1
            while d < n {
                for i in 0..<d {
                    let ai = offset * (2*i + 1) - 1
                    let bi = offset * (2*i + 2) - 1

                    let t = a[ai]
                    a[ai] = a[bi]
                    a[bi] = t + a[bi]
                }
                d *= 2
                offset >>= 1
            }
        }

        let n = 1024
        let data = Array(0..<n)
        var arr = data

        blellochScan(&arr)

        print(arr)
    }

}
