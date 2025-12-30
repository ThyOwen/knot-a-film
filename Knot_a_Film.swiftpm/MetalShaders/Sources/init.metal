#if defined(__METAL_VERSION__)

#include <metal_stdlib>
#include "MetalShaders.h"

using namespace metal;

#define ELEMENTS_PER_THREAD 4

//this computes the prefix sums from the offsets and then uses those to write connections to device memory
kernel void initalizeConnections( const device uint* __restrict connections [[buffer(0)]],
                                  const device uint* __restrict offsets [[buffer(1)]],
                                 
                                  device BodyMemberData<uint>& bodyOffsets [[buffer(BODY_OFFSETS_IDX)]], //should probably by intialized directly 
                                  device ConnectionsData& connectionsData [[buffer(CONNECTIONS_IDX)]],
                              
                                  threadgroup int* __restrict__ shared [[threadgroup(0)]],
                                  threadgroup int* __restrict__ simdSums [[threadgroup(1)]],
                              
                                  ushort tid [[thread_index_in_threadgroup]],
                                  ushort lane [[thread_index_in_simdgroup]],
                                  ushort simd_id [[simdgroup_index_in_threadgroup]],
                                  ushort tg_id [[threadgroup_position_in_grid]],
                                  ushort simdgroups_per_threadgroup [[simdgroups_per_threadgroup]],
                                  ushort threads_per_threadgroup [[threads_per_threadgroup]]
) {

    const uint base = tg_id * (threads_per_threadgroup * ELEMENTS_PER_THREAD);
    const uint local_base = tid * ELEMENTS_PER_THREAD;

    // Load and sum elements from offsets
    int vals[ELEMENTS_PER_THREAD];
    int sum = 0;
    
    for (short i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        uint idx = base + local_base + i;
        vals[i] = (idx < connectionsData.numBodies) ? offsets[idx] : 0;
        sum += vals[i];
    }

    // Compute prefix sum within simdgroup
    int simd_offset = simd_prefix_exclusive_sum(sum);

    int running = simd_offset;
    for (short i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        shared[local_base + i] = running;
        running += vals[i];
    }

    int simd_total = simd_sum(sum);

    if (lane == 0) {
        simdSums[simd_id] = simd_total;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scan simdgroup sums
    if (simd_id == 0) {
        int val = (lane < simdgroups_per_threadgroup) ? simdSums[lane] : 0;
        int scanned = simd_prefix_exclusive_sum(val);

        if (lane < simdgroups_per_threadgroup) {
            simdSums[lane] = scanned;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Add prefix of tile sums and write results
    int carry = simdSums[simd_id];

    for (short i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        uint local_idx = local_base + i;
        uint global_idx = base + local_idx;

        if (global_idx < connectionsData.numBodies) {
            int prefixSum = shared[local_idx] + carry;
            bodyOffsets.data[global_idx] = prefixSum;
            
            uint startIdx = prefixSum;
            uint count = vals[i];
            
            for (int k = 0; k < (int)count; ++k) {
                connectionsData.connections[startIdx + k] = connections[startIdx + k];
            }
        }
    }
}

kernel void initalizeBodies( device BodyMemberData<float>& mass [[buffer(BODY_MASS_IDX)]],
                             device BodyMemberData<float>& radius [[buffer(BODY_RADIUS_IDX)]],
                             device BodyMemberData<float2>& position [[buffer(BODY_POSITION_IDX)]],
                             device BodyMemberData<float2>& velocity [[buffer(BODY_VELOCITY_IDX)]],
                             device BodyMemberData<float2>& acceleration [[buffer(BODY_ACCELERATION_IDX)]],
                             device BodyMemberData<uint>& initialIdx [[buffer(BODY_INITIAL_IDX_IDX)]],
                             constant PhysicsParams& params [[buffer(CONNECTIONS_IDX)]],
                             uint gid [[thread_position_in_grid]]
) {
    if (gid >= numBodies) {
        return;
    }

    float maxDistance = 0.8f;
    float minDistance = 0.3f;
    float2 centerPos = { 0.0h, 0.0h };

    float angle = 2.0f * M_PI_F * (float(gid) / (float)(numBodies - 1));
    float radiusVal = (maxDistance - minDistance) * (fract(sin(float(gid) * 12.9898f) * 43758.5453f)) + minDistance;

    float x = centerPos.x + radiusVal * cos(angle);
    float y = centerPos.y + radiusVal * sin(angle);

    mass.data[gid] = 1.0f;
    radius.data[gid] = 0.001f;
    position.data[gid] = { x, y };
    velocity.data[gid] = { 0.0f, 0.0f };
    acceleration.data[gid] = { 0.0f, 0.0f };
    initialIdx.data[gid] = gid;
}


#endif

