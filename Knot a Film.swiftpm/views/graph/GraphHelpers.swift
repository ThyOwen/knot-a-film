//
//  GraphHelpers.swift
//  MetalTest
//
//  Created by Owen O'Malley on 8/27/25.
//

import Foundation
import MetalPerformanceShadersGraph

extension Graph {
    
    static func formatValue<T: Numeric>(_ value: T) -> String {
        switch value {
        case let v as Float:  return String(format: "%7.3f", v)
        case let v as Double: return String(format: "%7.3f", v)
        case let v as Int:    return String(format: "%7d", v)
        case let v as Int32:  return String(format: "%7d", v)
        case let v as Int64:  return String(format: "%7ld", v)
        default:              return "\(value)"
        }
    }
    
    internal mutating func recursivePrint(
        shape: [Int],
        offset: Int,
        stride: Int
    ) {
        if shape.count == 1 {
            let row = (0..<shape[0]).map { i in
                Self.formatValue(self.scratchBuffer[offset + i * stride])
            }
            print(row.joined(separator: "\t"))
        } else {
            let subShape = Array(shape.dropFirst())
            let innerStride = stride * subShape.reduce(1, *)
            for i in 0..<shape[0] {
                print("Slice \(i):")
                self.recursivePrint(shape: subShape, offset: offset + i * innerStride, stride: stride)
                print("")
            }
        }
    }

    @inlinable static func getNumRelevantNodes(numNodes: Int32) -> NSNumber {
        return NSNumber(value: numNodes * (numNodes - 1) / 2)
    }

    @inlinable static func getFlattenedSize(from shape : borrowing [NSNumber]) -> Int {
        return shape.reduce(1) { (currentProduct, element) in
            return currentProduct * element.intValue
        }
    }
    
}

extension GraphRenderer {
    static func buildStructuredPositions<N: FloatingPoint & SIMDScalar>(
        numNodes: Int,
        of type: N.Type,
        numSteps: Int = 4
    ) -> [SIMD2<N>] {
        let positions: [SIMD2<N>] = .init(unsafeUninitializedCapacity: numNodes) { buffer, initializedCount in
            let max: N = 1
            let gridSize = numSteps * numSteps  // number of regions
            let nodesPerRegion = (numNodes + gridSize - 1) / gridSize  // ceil division

            var idx = 0
            for region in 0..<gridSize {
                let gx = region % numSteps
                let gy = region / numSteps

                for j in 0..<nodesPerRegion {
                    guard idx < numNodes else { break }

                    // Spread nodes inside each region slightly
                    let localX: N = N(j % nodesPerRegion) * (max / N(nodesPerRegion))
                    let localY: N = N(j / nodesPerRegion) * (max / N(nodesPerRegion))

                    // Region center offset
                    let x = (N(gx) - N(numSteps - 1) / 2) * max + localX
                    let y = (N(gy) - N(numSteps - 1) / 2) * max + localY

                    buffer[idx] = SIMD2<N>(x, y)
                    idx += 1
                    //print(x,y)
                }
            }
            initializedCount = numNodes
        }
        return positions
    }
    
    static func buildRandomPositions<N: FloatingPoint & SIMDScalar>(numNodes: Int, of type: N.Type) -> [SIMD2<N>] {
        let positions: [SIMD2<N>] = .init(unsafeUninitializedCapacity: numNodes) { buffer, initializedCount in
            for idx in 0..<numNodes {

                if N.self == Float32.self {
                    let x = Float32.random(in: -0.5...0.5) as! N
                    let y = Float32.random(in: -0.5...0.5) as! N
                    
                    buffer[idx] = .init(x: x, y: y)
                } else if N.self == Float16.self {
                    let x = Float16.random(in: -0.5...0.5) as! N
                    let y = Float16.random(in: -0.5...0.5) as! N
                    
                    buffer[idx] = .init(x: x, y: y)
                } else {
                    fatalError("ASHDFASJDHFLAJKSDFLKAJSDFLKAJSDF")
                }
            }
            initializedCount = numNodes
        }
        return positions
    }
}
 
extension Numeric {
    @inlinable static func mpsDataType() -> MPSDataType {
        switch Self.self {
        case is Int8.Type:
            return .int8
        case is UInt8.Type:
            return .uInt8
        case is Int16.Type:
            return .int16
        case is UInt16.Type:
            return .uInt16
        case is Int32.Type:
            return .int32
        case is UInt32.Type:
            return .uInt32
        case is Int64.Type:
            return .int64
        case is UInt64.Type:
            return .uInt64
        case is Float32.Type:
            return .float32
        case is Float16.Type:
            return .float16
        default:
            fatalError("Unsupported numeric type in MPS type: \(Self.self)")
        }
    }
}

extension MPSDataType : @retroactive CustomStringConvertible {
    public var description: String {
        switch self {
        case .float16: return "float16"
        case .float32: return "float32"
        case .int8: return "int8"
        case .int16: return "int16"
        case .int32: return "int32"
        case .int64: return "int64"
        case .uInt8: return "uInt8"
        case .uInt16: return "uInt16"
        case .uInt32: return "uInt32"
        case .uInt64: return "uInt64"
        case .invalid: return "invalid"
        default: return "other"
        }
    }
}
