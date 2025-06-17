//
//  Edge+SIMD.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 2/6/25.
//

extension MovieEdge {
    public func withBufferAccess(to array: borrowing ContiguousArray<N>, operation: (_ aNode : consuming N,
                                                                                     _ bNode : consuming N) -> Void) {

        array.withContiguousStorageIfAvailable { arrayPointer in
            operation(arrayPointer[self.aNodePositionIndex], arrayPointer[self.bNodePositionIndex])
        }
    }
    
    public func withBufferAccess(to array: inout ContiguousArray<N>, operation: (_ aNode : inout N,
                                                                                 _ bNode : inout N) -> Void) {

        array.withContiguousMutableStorageIfAvailable { arrayPointer in
            operation(&arrayPointer[self.aNodePositionIndex], &arrayPointer[self.bNodePositionIndex])
        }
    }
}
