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
