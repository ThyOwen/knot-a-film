//
//  forces.metal
//  Knot a Film
//
//  Created by Owen O'Malley on 12/2/25.
//

kernel void computeForces(device const float* inA,
                          device const float* inB,
                          device float* result,
                          uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] + inB[index];
}
