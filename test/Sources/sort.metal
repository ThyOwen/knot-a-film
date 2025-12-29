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
#define ELEMENTS_PER_SIMD (SIMD_SIZE * ELEMENTS_PER_THREAD)
#define TILES_PER_THREADGROUP (ARRAY_SIZE / ELEMENTS_PER_SIMD)

kernel void prefixSum(device const int* inData [[buffer(1)]],
                      device int* outData [[buffer(2)]],
                      constant uint& n [[buffer(0)]],
                      uint simdgroup_id [[simdgroup_index_in_threadgroup]],
                      uint simd_lane_id [[thread_index_in_simdgroup]],
                      uint threads_per_threadgroup [[threads_per_threadgroup]],
                      uint threadgroup_position_in_grid [[threadgroup_position_in_grid]]
) {
    threadgroup int sharedData[ARRAY_SIZE];
    threadgroup int tileSums[TILES_PER_THREADGROUP];
    
    const uint simdgroups_per_threadgroup = threads_per_threadgroup / SIMD_SIZE;
    const uint tiles_per_simdgroup = TILES_PER_THREADGROUP / simdgroups_per_threadgroup;
    const uint base_offset = threadgroup_position_in_grid * ARRAY_SIZE;
    
    // simdgroup assigned tiles
    for (uint tile = 0; tile < tiles_per_simdgroup; tile++) {
        uint tile_idx = simdgroup_id * tiles_per_simdgroup + tile;
        uint tile_offset = tile_idx * ELEMENTS_PER_SIMD;
        
        int values[ELEMENTS_PER_THREAD];
        int thread_sum = 0;
        
        for (uint i = 0; i < ELEMENTS_PER_THREAD; i++) {
            uint global_idx = base_offset + tile_offset + simd_lane_id * ELEMENTS_PER_THREAD + i;
            values[i] = (global_idx < n) ? inData[global_idx] : 0;
            thread_sum += values[i];
        }
        
        int thread_offset = simd_prefix_exclusive_sum(thread_sum);
        
        int running_sum = thread_offset;
        for (uint i = 0; i < ELEMENTS_PER_THREAD; i++) {
            uint local_idx = tile_offset + simd_lane_id * ELEMENTS_PER_THREAD + i;
            sharedData[local_idx] = running_sum;
            running_sum += values[i];
        }
        
        int tile_sum = simd_sum(thread_sum);
        
        if (simd_lane_id == 0) {
            tileSums[tile_idx] = tile_sum;
        }
        
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // scan the tile sums to get prefix sums
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
    
    // add prefix of tile sums to each element and write out
    for (uint tile = 0; tile < tiles_per_simdgroup; tile++) {
        uint tile_idx = simdgroup_id * tiles_per_simdgroup + tile;
        uint tile_offset = tile_idx * ELEMENTS_PER_SIMD;
        
        int carry = tileSums[tile_idx];
        
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

#define THREADS_PER_THREADGROUP 32
#define SIMDGROUPS_PER_TG (THREADS_PER_THREADGROUP / SIMD_SIZE)
#define ELEMENTS_PER_TG (THREADS_PER_THREADGROUP * ELEMENTS_PER_THREAD)

kernel void prefixGlobalSum( device const int* inData [[buffer(0)]],
                             device int*       outData [[buffer(1)]],
                             constant uint&    n [[buffer(2)]],

                             uint tid [[thread_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simd_id  [[simdgroup_index_in_threadgroup]],
                             uint tg_id  [[threadgroup_position_in_grid]]
)
{
    threadgroup int shared[ELEMENTS_PER_TG];
    threadgroup int simdSums[SIMDGROUPS_PER_TG];

    const uint base = tg_id * ELEMENTS_PER_TG;
    const uint local_base = tid * ELEMENTS_PER_THREAD;

    // simdgroup assigned tiles
    int vals[ELEMENTS_PER_THREAD];
    int sum = 0;

    for (uint i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        uint idx = base + local_base + i;
        vals[i] = (idx < n) ? inData[idx] : 0;
        sum += vals[i];
    }

    int simd_offset = simd_prefix_exclusive_sum(sum);

    int running = simd_offset;
    for (uint i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        shared[local_base + i] = running;
        running += vals[i];
    }

    int simd_total = simd_sum(sum);

    if (lane == 0) {
        simdSums[simd_id] = simd_total;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // scan the tile sums to get prefix sums
    if (simd_id == 0) {
        int val = (lane < SIMDGROUPS_PER_TG) ? simdSums[lane] : 0;
        int scanned = simd_prefix_exclusive_sum(val);

        if (lane < SIMDGROUPS_PER_TG) {
            simdSums[lane] = scanned;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // add prefix of tile sums to each element and write out
    int carry = simdSums[simd_id];

    for (uint i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        uint local_idx = local_base + i;
        uint global_idx = base + local_idx;

        if (global_idx < n) {
            outData[global_idx] = shared[local_idx] + carry;
        }
    }
}
