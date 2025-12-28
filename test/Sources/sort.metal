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
#define ELEMENTS_PER_THREAD 4
#define ELEMENTS_PER_TILE (SIMD_SIZE * ELEMENTS_PER_THREAD)
#define TILES_PER_THREADGROUP (ARRAY_SIZE / ELEMENTS_PER_TILE)

kernel void prefixSum(constant uint& n [[buffer(0)]],
                      device const int* inData [[buffer(1)]],
                      device int* outData [[buffer(2)]],
                      uint simdgroup_id [[simdgroup_index_in_threadgroup]],
                      uint simd_lane_id [[thread_index_in_simdgroup]],
                      uint threads_per_threadgroup [[threads_per_threadgroup]],
                      uint threadgroup_position_in_grid [[threadgroup_position_in_grid]])
{
    threadgroup int sharedData[ARRAY_SIZE];
    threadgroup int tileSums[TILES_PER_THREADGROUP];
    
    const uint simdgroups_per_threadgroup = threads_per_threadgroup / SIMD_SIZE;
    const uint tiles_per_simdgroup = TILES_PER_THREADGROUP / simdgroups_per_threadgroup;
    const uint base_offset = threadgroup_position_in_grid * ARRAY_SIZE;
    
    // Phase 1: Each simdgroup loads and processes its assigned tiles
    for (uint tile = 0; tile < tiles_per_simdgroup; tile++) {
        uint tile_idx = simdgroup_id * tiles_per_simdgroup + tile;
        uint tile_offset = tile_idx * ELEMENTS_PER_TILE;
        
        // Load ELEMENTS_PER_THREAD values per thread
        int values[ELEMENTS_PER_THREAD];
        int thread_sum = 0;
        
        for (uint i = 0; i < ELEMENTS_PER_THREAD; i++) {
            uint global_idx = base_offset + tile_offset + simd_lane_id * ELEMENTS_PER_THREAD + i;
            values[i] = (global_idx < n) ? inData[global_idx] : 0;
            thread_sum += values[i];
        }
        
        // Step 1: Exclusive scan across threads to get per-thread offsets
        int thread_offset = simd_prefix_exclusive_sum(thread_sum);
        
        // Step 2: Sequential scan within each thread's elements
        int running_sum = thread_offset;
        for (uint i = 0; i < ELEMENTS_PER_THREAD; i++) {
            uint local_idx = tile_offset + simd_lane_id * ELEMENTS_PER_THREAD + i;
            sharedData[local_idx] = running_sum;
            running_sum += values[i];
        }
        
        // Compute tile sum (sum of all elements in tile)
        int tile_sum = simd_sum(thread_sum);
        
        if (simd_lane_id == 0) {
            tileSums[tile_idx] = tile_sum;
        }
        
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Phase 2: Scan the tile sums (unchanged)
    if (simdgroup_id == 0) {
        int running_sum = 0;
        
        for (uint i = 0; i < TILES_PER_THREADGROUP; i += SIMD_SIZE) {
            uint idx = i + simd_lane_id;
            int val = (idx < TILES_PER_THREADGROUP) ? tileSums[idx] : 0;
            
            int scanned = simd_prefix_exclusive_sum(val) + running_sum;
            
            if (idx < TILES_PER_THREADGROUP) {
                tileSums[idx] = scanned;
            }
            
            running_sum += simd_sum(val);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Phase 3: Add prefix of tile sums and write out
    for (uint tile = 0; tile < tiles_per_simdgroup; tile++) {
        uint tile_idx = simdgroup_id * tiles_per_simdgroup + tile;
        uint tile_offset = tile_idx * ELEMENTS_PER_TILE;
        
        int carry = tileSums[tile_idx];
        
        // Write ELEMENTS_PER_THREAD values per thread
        for (uint i = 0; i < ELEMENTS_PER_THREAD; i++) {
            uint local_idx = tile_offset + simd_lane_id * ELEMENTS_PER_THREAD + i;
            uint global_idx = base_offset + local_idx;
            
            if (global_idx < n) {
                outData[global_idx] = sharedData[local_idx] + carry;
            }
        }
        
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}
