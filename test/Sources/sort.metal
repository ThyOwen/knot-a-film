#include <metal_stdlib>
using namespace metal;

constant int ITEMS_PER_THREAD = 32; // 1024 items / 32 threads
constant int BUFFER_SIZE = 1024;

kernel void radixSort(device uint* input [[buffer(0)]],
                      device uint* output [[buffer(1)]],
                      constant uint& num [[buffer(2)]],
                      ushort lane_id [[thread_index_in_threadgroup]],
                      ushort simd_size [[threads_per_simdgroup]]
) {
    threadgroup uint sharedData[2 * BUFFER_SIZE];
        
    // load everything
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        uint idx = lane_id + i * simd_size;
        if (idx < num) {
            sharedData[idx] = input[idx];
        } else {
            sharedData[idx] = 0xFFFFFFFF; // 0xFFFFFFFF (max uint) so they sort to the end naturally
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    uint srcOffset = 0;
    uint dstOffset = BUFFER_SIZE;
    
    // sort loop for 32 bits
    for (int bit = 0; bit < 32; bit++) {
        
        // count zeros
        uint totalZeros = 0;
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            uint val = sharedData[srcOffset + lane_id + i * simd_size];
            bool isZero = !((val >> bit) & 1);
            
            totalZeros += simd_sum(isZero ? 1 : 0); // a 1 is a partition point
        }
        
        uint runningZeros = 0;
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            uint idx = lane_id + i * simd_size;
            uint val = sharedData[srcOffset + idx];
            bool isZero = !((val >> bit) & 1);
            
            // calculate rank of this element within the current row (0..31)
            uint localRank = simd_prefix_exclusive_sum(isZero ? 1 : 0);
            
            // broadcast the number of zeros of this row which lives in the last lane
            uint rowZeroCount = simd_broadcast(localRank + (isZero ? 1 : 0), 31);
            
            uint targetIdx;
            if (isZero) {
                targetIdx = runningZeros + localRank; // write at (zeros seen previously) + (zeros to my left)
            } else {
                targetIdx = totalZeros + (idx - (runningZeros + localRank));
            }
            
            sharedData[dstOffset + targetIdx] = val;

            runningZeros += rowZeroCount; // advance the running count for the next row
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // swap buffers
        srcOffset = BUFFER_SIZE - srcOffset;
        dstOffset = BUFFER_SIZE - dstOffset;
    }
    
    // store result
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        uint idx = lane_id + i * simd_size;
        if (idx < num) {
            output[idx] = sharedData[srcOffset + idx];
        }
    }
}

uint floatToUint(float f) {
    uint u = as_type<uint>(f);
    uint mask = (u >> 31) ? 0xFFFFFFFF : 0x80000000;
    return u ^ mask;
}

float uintToFloat(uint u) {
    uint mask = (u >> 31) ? 0x80000000 : 0xFFFFFFFF;
    return as_type<float>(u ^ mask);
}

kernel void radixSortKV(device float* keysIn     [[buffer(0)]],
                        device uint* valuesIn   [[buffer(1)]],
                        device float* keysOut    [[buffer(2)]],
                        device uint* valuesOut  [[buffer(3)]],
                        constant uint& num       [[buffer(4)]],
                        ushort lane_id           [[thread_index_in_threadgroup]],
                        ushort simd_size         [[threads_per_simdgroup]])
{
    // ping_pongOffsetset + element_index
    threadgroup uint sharedKeys[2 * BUFFER_SIZE];
    threadgroup uint sharedValues[2 * BUFFER_SIZE];

    // load everything
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        uint idx = lane_id + i * simd_size;
        if (idx < num) {
            sharedKeys[idx] = floatToUint(keysIn[idx]);
            sharedValues[idx] = valuesIn[idx];
        } else {
            sharedKeys[idx] = 0xFFFFFFFF; // 0xFFFFFFFF (max uint) so they sort to the end naturally
            sharedValues[idx] = 0;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    uint srcOffset = 0;
    uint dstOffset = BUFFER_SIZE;
    
    // sort loop for 32 bits
    for (int bit = 0; bit < 32; bit++) {
        
        // count zeros
        uint totalZeros = 0;
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            uint val = sharedKeys[srcOffset + lane_id + i * simd_size];
            bool isZero = !((val >> bit) & 1);
            totalZeros += simd_sum(isZero ? 1 : 0); // a 1 is a partition point
        }
        
        // scatter Keys and Values
        uint runningZeros = 0;
        
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            uint idx = lane_id + i * simd_size;
            uint key = sharedKeys[srcOffset + idx];
            uint val = sharedValues[srcOffset + idx];
            
            bool isZero = !((key >> bit) & 1);
            
            // calculate rank of this element within the current row (0..31)
            uint localRank = simd_prefix_exclusive_sum(isZero ? 1 : 0);
            
            // broadcast the number of zeros of this row which lives in the last lane
            uint rowZeroCount = simd_broadcast(localRank + (isZero ? 1 : 0), 31);
            
            uint targetIdx;
            if (isZero) {
                targetIdx = runningZeros + localRank; // write at (zeros seen previously) + (zeros to my left)
            } else {
                targetIdx = totalZeros + (idx - (runningZeros + localRank));
            }
            
            sharedKeys[dstOffset + targetIdx] = key;
            sharedValues[dstOffset + targetIdx] = val;
            
            runningZeros += rowZeroCount; // advance the running count for the next row
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // swap offsets
        srcOffset = BUFFER_SIZE - srcOffset;
        dstOffset = BUFFER_SIZE - dstOffset;
    }
    
    // store result
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        uint idx = lane_id + i * simd_size;
        if (idx < num) {
            keysOut[idx] = uintToFloat(sharedKeys[srcOffset + idx]);
            valuesOut[idx] = sharedValues[srcOffset + idx];
        }
    }
}
