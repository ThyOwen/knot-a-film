//
//  sort.metal
//  MetalTest
//
//  Created by Owen O'Malley on 12/27/25.
//

#include <metal_stdlib>

using namespace metal;

#define ARRAY_SIZE 1024
#define SIMD_SIZE 32
#define TILES_PER_SIMDGROUP (ARRAY_SIZE / SIMD_SIZE)

kernel void prefixSumSimple( constant int& n [[buffer(0)]],
                             device int* data [[buffer(1)]],
                             device int* outData [[buffer(2)]],
                             ushort simd_lane_id [[thread_index_in_simdgroup]]
) {
    threadgroup float temp[SIMD_SIZE];
    
    int offset = 1;
    
    temp[2 * simd_lane_id + 1] = data[2 * simd_lane_id];
    
    for (int d = n >> 1; d > 0; d >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_lane_id < d) {
            int ai = offset * (2*simd_lane_id + 1) - 1;
            int bi = offset * (2*simd_lane_id + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    if (simd_lane_id == 0) {
        temp[n - 1] = 0;
    }
    
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane_id < d) {
            int ai = offset * (2 * simd_lane_id + 1) - 1;
            int bi = offset * (2 * simd_lane_id + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    outData[2 * simd_lane_id] = temp[2 * simd_lane_id]; // write results to device memory
    outData[2 * simd_lane_id + 1] = temp[2 * simd_lane_id + 1];

}

kernel void prefixSum(constant int& n [[buffer(0)]],
                      device const int* inData [[buffer(1)]],
                      device int* outData [[buffer(2)]],
                      uint simdgroup_id [[simdgroup_index_in_threadgroup]],
                      uint simd_lane_id [[thread_index_in_simdgroup]],
                      uint threads_per_threadgroup [[threads_per_threadgroup]],
                      uint threadgroup_position_in_grid [[threadgroup_position_in_grid]])
{
    threadgroup int sharedData[ARRAY_SIZE];
    threadgroup int tileSums[TILES_PER_SIMDGROUP];
    
    const int simdgroups_per_threadgroup = threads_per_threadgroup / SIMD_SIZE;
    const int tiles_per_simdgroup = TILES_PER_SIMDGROUP / simdgroups_per_threadgroup;
    const int base_offset = threadgroup_position_in_grid * ARRAY_SIZE;
    
    // simdgroup assigned tiles
    for (int tile = 0; tile < tiles_per_simdgroup; tile++) {
        int tile_idx = simdgroup_id * tiles_per_simdgroup + tile;
        int tile_offset = tile_idx * SIMD_SIZE;
        int global_idx = base_offset + tile_offset + simd_lane_id;
        
        int value = (global_idx < n) ? inData[global_idx] : 0;
        
        int scanned = simd_prefix_exclusive_sum(value);
        sharedData[tile_offset + simd_lane_id] = scanned;
        
        int tile_sum = simd_sum(value);
        
        if (simd_lane_id == 0) {
            tileSums[tile_idx] = tile_sum;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // scan the tile sums to get prefix sums
    if (simdgroup_id == 0) {
        int running_sum = 0;
        
        for (int i = 0; i < TILES_PER_SIMDGROUP; i += SIMD_SIZE) {
            uint idx = i + simd_lane_id;
            int val = (idx < TILES_PER_SIMDGROUP) ? tileSums[idx] : 0;
            
            int scanned = simd_prefix_exclusive_sum(val) + running_sum;
            
            if (idx < TILES_PER_SIMDGROUP) {
                tileSums[idx] = scanned;
            }
            
            running_sum += simd_sum(val);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // add prefix of tile sums to each element and write out
    for (int tile = 0; tile < tiles_per_simdgroup; tile++) {
        int tile_idx = simdgroup_id * tiles_per_simdgroup + tile;
        int tile_offset = tile_idx * SIMD_SIZE;
        int global_idx = base_offset + tile_offset + simd_lane_id;
        
        int carry = tileSums[tile_idx];
        int final_value = sharedData[tile_offset + simd_lane_id] + carry;
        
        if (global_idx < n) {
            outData[global_idx] = final_value;
        }
    }
}
