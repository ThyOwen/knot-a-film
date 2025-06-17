//
//  Graph+SIMD.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 2/2/25.
//

import Accelerate

extension GraphManager {
    
    internal static func withSIMDAccess<T : MutableCollection>(to array : inout T, operation : (inout SIMDType) -> Void) where T.Element == Double {
        array.withContiguousMutableStorageIfAvailable { arrayPointer in
            guard let baseAddress = arrayPointer.baseAddress else { return }

            baseAddress.withMemoryRebound(to: SIMDType.self, capacity: 1) { SIMDPointer in
                operation(&SIMDPointer.pointee)
            }
        }
    }
    
    internal static func withSIMDAccess<T : Collection>(to array : borrowing T, operation : (consuming SIMDType) -> Void) where T.Element == Double {
        array.withContiguousStorageIfAvailable { arrayPointer in
            guard let baseAddress = arrayPointer.baseAddress else { return }
            baseAddress.withMemoryRebound(to: SIMDType.self, capacity: 1) { SIMDPointer in
                operation(SIMDPointer.pointee)
            }
        }
    }
    
    
    internal static func withRangedSIMDAccess(to array : inout ContiguousArray<Double>,
                                             aRange: Range<Int>,
                                             bRange: Range<Int>,
                                             operation : (_ aNode : inout SIMDType, _ bNode : inout SIMDType) -> Void) {
        array.withContiguousMutableStorageIfAvailable { arrayPointer in
            arrayPointer[aRange].withContiguousMutableStorageIfAvailable { aPointer in
                arrayPointer[bRange].withContiguousMutableStorageIfAvailable { bPointer in
                    
                    guard let aBaseAddress = aPointer.baseAddress else { return }
                    guard let bBaseAddress = bPointer.baseAddress else { return }
                    
                    aBaseAddress.withMemoryRebound(to: SIMDType.self, capacity: 1) { aSIMDPointer in
                        bBaseAddress.withMemoryRebound(to: SIMDType.self, capacity: 1) { bSIMDPointer in
                            operation(&aSIMDPointer.pointee, &bSIMDPointer.pointee)
                        }
                    }
                }
            }
        }
        
    }
    
    internal static func withRangedSIMDAccess(to array : borrowing ContiguousArray<Double>,
                                             aRange: Range<Int>,
                                             bRange: Range<Int>,
                                             operation : (_ aNode : consuming SIMDType, _ bNode : consuming SIMDType) -> Void) {
        array.withContiguousStorageIfAvailable { arrayPointer in
            arrayPointer[aRange].withContiguousStorageIfAvailable { aPointer in
                arrayPointer[bRange].withContiguousStorageIfAvailable { bPointer in
                    
                    guard let aBaseAddress = aPointer.baseAddress else { return }
                    guard let bBaseAddress = bPointer.baseAddress else { return }
                    
                    aBaseAddress.withMemoryRebound(to: SIMDType.self, capacity: 1) { aSIMDPointer in
                        bBaseAddress.withMemoryRebound(to: SIMDType.self, capacity: 1) { bSIMDPointer in
                            operation(aSIMDPointer.pointee, bSIMDPointer.pointee)
                        }
                    }
                }
            }
        }
        
    }
    
}

extension GraphManager {
    internal static func withRangedMutableAccess(to buffer : borrowing ManagedBuffer<Int, Double>,
                                                 aRange: Range<Int>,
                                                 bRange: Range<Int>,
                                                 operation : (_ aNodeBaseAddress : consuming UnsafeMutablePointer<Double>,
                                                              _ bNodeBaseAddress : consuming UnsafeMutablePointer<Double>) -> Void) {
        buffer.withUnsafeMutablePointerToElements { bufferPointer in
            withUnsafeMutablePointer(to: &bufferPointer[aRange.lowerBound]) { aPointer in
                withUnsafeMutablePointer(to: &bufferPointer[bRange.lowerBound]) { bPointer in
                    operation(aPointer, bPointer)
                }
            }
        }
    }
    
    internal static func withRangedAccess(to buffer : borrowing ManagedBuffer<Int, Double>,
                                          aRange: Range<Int>,
                                          bRange: Range<Int>,
                                          operation : (_ aNodeBaseAddress : consuming UnsafePointer<Double>,
                                                       _ bNodeBaseAddress : consuming UnsafePointer<Double>) -> Void) {
        buffer.withUnsafeMutablePointerToElements { bufferPointer in
            withUnsafePointer(to: bufferPointer[aRange.lowerBound]) { aPointer in
                withUnsafePointer(to: bufferPointer[bRange.lowerBound]) { bPointer in
                    operation(aPointer, bPointer)
                }
            }
        }
    }
    
}
