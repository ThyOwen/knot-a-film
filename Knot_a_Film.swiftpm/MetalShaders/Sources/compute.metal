#if defined(__METAL_VERSION__)

#include <metal_stdlib>
#include "MetalShaders.h"

using namespace metal;

kernel void resetKernel(device NodeMemberData<float2>& topLeft  [[buffer(NODE_TOP_LEFT_IDX)]],
                        device NodeMemberData<float2>& bottomRight  [[buffer(NODE_BOTTOM_RIGHT_IDX)]],
                        device NodeMemberData<float2>& centerOfMass  [[buffer(NODE_CENTER_OF_MASS_IDX)]],
                        device NodeMemberData<float>& totalMass  [[buffer(NODE_TOTAL_MASS_IDX)]],
                        device NodeMemberData<uint>& start  [[buffer(NODE_START_IDX)]],
                        device NodeMemberData<uint>& end  [[buffer(NODE_END_IDX)]],
                        device NodeMemberData<bool>& isLeaf  [[buffer(NODE_IS_LEAF_IDX)]],
                        device atomic_int *mutex  [[buffer(MUTEX_IDX)]],
                        ushort tid [[thread_position_in_threadgroup]],
                        uint bid [[threadgroup_position_in_grid]],
                        ushort simd_lane_id [[thread_index_in_simdgroup]],
                        ushort threads_per_threadgroup [[threads_per_threadgroup]]
) {
    uint gid = bid * threads_per_threadgroup + tid;
    
    if (gid < uint(topLeft.size)) {
        topLeft.data[gid] = {INFINITY, -INFINITY};
        bottomRight.data[gid] = {-INFINITY, INFINITY};
        centerOfMass.data[gid] = {-1.0h, -1.0h};
        totalMass.data[gid] = 0.0;
        isLeaf.data[gid] = true;
        start.data[gid] = -1;
        end.data[gid] = -1;
        atomic_store_explicit(&mutex[gid], 0, memory_order_relaxed);
    }

    if (bid == 0 && tid == 0) {
        start.data[0] = 0;
        end.data[0] = numBodies - 1;
    }
}

kernel void computeBoundingBoxKernel(device NodeMemberData<float2>& topLeft [[buffer(NODE_TOP_LEFT_IDX)]],
                                     device NodeMemberData<float2>& bottomRight [[buffer(NODE_BOTTOM_RIGHT_IDX)]],
                                     constant BodyMemberData<float2>& position [[buffer(BODY_POSITION_IDX)]],
                                     device atomic_int *mutex [[buffer(MUTEX_IDX)]],
                                     ushort tid [[thread_position_in_threadgroup]],
                                     uint bid [[threadgroup_position_in_grid]],
                                     ushort simd_lane_id [[thread_index_in_simdgroup]],
                                     ushort threads_per_threadgroup [[threads_per_threadgroup]]
) {
    uint gid = bid * threads_per_threadgroup + tid;
    
    float topLeftX = INFINITY;
    float topLeftY = -INFINITY;
    float bottomRightX = -INFINITY;
    float bottomRightY = INFINITY;

    if (gid < (uint)numBodies) {
        float2 pos = position.data[gid];
        topLeftX = pos.x;
        topLeftY = pos.y;
        bottomRightX = pos.x;
        bottomRightY = pos.y;
    }

    topLeftX = simd_min(topLeftX);
    topLeftY = simd_max(topLeftY);
    bottomRightX = simd_max(bottomRightX);
    bottomRightY = simd_min(bottomRightY);

    if (simd_lane_id == 0) {
        int expected = 0;
        while (!atomic_compare_exchange_weak_explicit(&mutex[0], &expected, 1,
                                                       memory_order_relaxed,
                                                       memory_order_relaxed))
        {
            expected = 0;
        }
        
        float rangeX = topLeftX - bottomRightX;
        float rangeY = topLeftY - bottomRightY;
        float paddingX = max(abs(rangeX) * 0.01f, 0.1f);
        float paddingY = max(abs(rangeY) * 0.01f, 0.1f);
        
        topLeft.data[0].x = -10.0f;
        topLeft.data[0].y = 10.0f;
        bottomRight.data[0].x = 10.0f;
        bottomRight.data[0].y = -10.0f;
        
        atomic_store_explicit(&mutex[0], 0, memory_order_relaxed);
    }
}

inline int getQuadrant(float2 topLeft,
                       float2 bottomRight,
                       float x,
                       float y
) {
    float midX = (topLeft.x + bottomRight.x) * 0.5f;
    float midY = (topLeft.y + bottomRight.y) * 0.5f;
    
    bool left  = (midX >= x);
    bool top   = (midY <= y);

    return left
        ? (top ? 2 : 3)
        : (top ? 1 : 4);
}

inline void computeCenterOfMass(uint nodeIndex,
                                device NodeMemberData<float2>& centerOfMass,
                                device NodeMemberData<float>& totalMass,
                                constant BodyMemberData<float2>& position,
                                constant BodyMemberData<float>& mass,
                                int start,
                                int end,
                                ushort tid,
                                ushort simd_lane_id,
                                ushort threads_per_threadgroup
) {
    int total = end - start + 1;
    int sz = (total + threads_per_threadgroup - 1) / threads_per_threadgroup;
    int s = tid * sz + start;
    float M = 0.0;
    float2 R = { 0.0f, 0.0f };

    for (int i = s; i < s + sz; ++i) {
        if (i <= end) {
            float bodyMass = mass.data[i];
            float2 bodyPos = position.data[i];
            M += bodyMass;
            R += bodyMass * bodyPos;
        }
    }

    M = simd_sum(M);
    R.x = simd_sum(R.x);
    R.y = simd_sum(R.y);

    if (simd_lane_id == 0 && M > 0.0f) {
        R /= M;
        totalMass.data[nodeIndex] = M;
        centerOfMass.data[nodeIndex] = R;
    }
}

inline void countBodies( constant BodyMemberData<float2>& position,
                         float2 topLeft,
                         float2 bottomRight,
                         threadgroup int* __restrict__ count,
                         int start,
                         int end,
                         ushort tid,
                         ushort threads_per_threadgroup
) {
    if (tid < 4)
        count[tid] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = start + tid; i <= end; i += threads_per_threadgroup) {
        float2 pos = position.data[i];
        int q = getQuadrant(topLeft, bottomRight, pos.x, pos.y);
        atomic_fetch_add_explicit((threadgroup atomic_int*)&count[q - 1], 1, memory_order_relaxed);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void computeOffset(threadgroup int* __restrict__ count,
                          int start,
                          ushort tid
) {
    if (tid < 4) {
        int offset = start;
        for (int i = 0; i < int(tid); ++i) {
            offset += count[i];
        }
        count[tid + 4] = offset;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void groupBodies(constant BodyMemberData<float>& mass_in,
                        constant BodyMemberData<float>& radius_in,
                        constant BodyMemberData<float2>& position_in,
                        constant BodyMemberData<float2>& velocity_in,
                        constant BodyMemberData<float2>& acceleration_in,
                        constant BodyMemberData<uint>& initialIdx_in,
                        constant BodyMemberData<uint>& offsets_in,
                        device BodyMemberData<float>& mass_out,
                        device BodyMemberData<float>& radius_out,
                        device BodyMemberData<float2>& position_out,
                        device BodyMemberData<float2>& velocity_out,
                        device BodyMemberData<float2>& acceleration_out,
                        device BodyMemberData<uint>& initialIdx_out,
                        device BodyMemberData<uint>& offsets_out,
                        float2 topLeft,
                        float2 bottomRight,
                        threadgroup int* __restrict__ count,
                        int start,
                        int end,
                        ushort tid,
                        ushort threads_per_threadgroup
) {
    threadgroup int* count2 = &count[4];
    for (int i = start + tid; i <= end; i += threads_per_threadgroup)
    {
        float2 pos = position_in.data[i];
        int q = getQuadrant(topLeft, bottomRight, pos.x, pos.y);
        int dest = atomic_fetch_add_explicit((threadgroup atomic_int*)&count2[q - 1], 1, memory_order_relaxed);
        
        mass_out.data[dest] = mass_in.data[i];
        radius_out.data[dest] = radius_in.data[i];
        position_out.data[dest] = position_in.data[i];
        velocity_out.data[dest] = velocity_in.data[i];
        acceleration_out.data[dest] = acceleration_in.data[i];
        initialIdx_out.data[dest] = initialIdx_in.data[i];
        offsets_out.data[dest] = offsets_in.data[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void constructQuadTreeKernel(device NodeMemberData<float2>& topLeft [[buffer(NODE_TOP_LEFT_IDX)]],
                                    device NodeMemberData<float2>& bottomRight [[buffer(NODE_BOTTOM_RIGHT_IDX)]],
                                    device NodeMemberData<float2>& centerOfMass [[buffer(NODE_CENTER_OF_MASS_IDX)]],
                                    device NodeMemberData<float>& totalMass [[buffer(NODE_TOTAL_MASS_IDX)]],
                                    device NodeMemberData<uint>& start [[buffer(NODE_START_IDX)]],
                                    device NodeMemberData<uint>& end [[buffer(NODE_END_IDX)]],
                                    device NodeMemberData<bool>& isLeaf [[buffer(NODE_IS_LEAF_IDX)]],
                                    
                                    constant BodyMemberData<float>& mass_in [[buffer(BODY_MASS_IDX)]],
                                    constant BodyMemberData<float2>& position_in [[buffer(BODY_POSITION_IDX)]],
                                    constant BodyMemberData<uint>& initialIdx_in [[buffer(BODY_INITIAL_IDX_IDX)]],
                                    
                                    device BodyMemberData<float>& mass_out [[buffer(BODY_MASS_ALT_IDX)]],
                                    device BodyMemberData<float2>& position_out [[buffer(BODY_POSITION_ALT_IDX)]],
                                    device BodyMemberData<uint>& initialIdx_out [[buffer(BODY_INITIAL_IDX_ALT_IDX)]],
                                    
                                    constant int &nodeOffset [[buffer(22)]],
                                    ushort tid [[thread_position_in_threadgroup]],
                                    uint bid [[threadgroup_position_in_grid]],
                                    ushort simd_lane_id [[thread_index_in_simdgroup]],
                                    ushort threads_per_threadgroup [[threads_per_threadgroup]]
) {
    
    threadgroup int count[8];
    
    uint nodeIndex = nodeOffset + bid;

    if (nodeIndex >= topLeft.size)
        return;

    int nodeStart = start.data[nodeIndex];
    int nodeEnd = end.data[nodeIndex];
    float2 nodeTopLeft = topLeft.data[nodeIndex];
    float2 nodeBottomRight = bottomRight.data[nodeIndex];

    if (nodeStart == -1 || nodeEnd == -1)
        return;

    computeCenterOfMass(nodeIndex, centerOfMass, totalMass, position_in, mass_in, nodeStart, nodeEnd, tid, simd_lane_id, threads_per_threadgroup);
    
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (nodeIndex >= uint(leafLimit) || nodeStart == nodeEnd) {
        for (int i = nodeStart; i <= nodeEnd; ++i) {
            mass_out.data[i] = mass_in.data[i];
            position_out.data[i] = position_in.data[i];
            initialIdx_out.data[i] = initialIdx_in.data[i];

        }
        return;
    }

    countBodies(position_in, nodeTopLeft, nodeBottomRight, count, nodeStart, nodeEnd, tid, threads_per_threadgroup);
    computeOffset(count, nodeStart, tid);

    //group bodies
    threadgroup int* count2 = &count[4];
    for (int i = nodeStart + tid; i <= nodeEnd; i += threads_per_threadgroup) {
        float2 pos = position_in.data[i];
        int q = getQuadrant(nodeTopLeft, nodeBottomRight, pos.x, pos.y);
        int dest = atomic_fetch_add_explicit((threadgroup atomic_int*)&count2[q - 1], 1, memory_order_relaxed);
        
        mass_out.data[dest] = mass_in.data[i];
        position_out.data[dest] = position_in.data[i];
        initialIdx_out.data[dest] = initialIdx_in.data[i];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 4) {
        uint quadIndex = tid + 1;
        uint childIndex = (nodeIndex * 4) + quadIndex;

        float midX = (nodeTopLeft.x + nodeBottomRight.x) / 2;
        float midY = (nodeTopLeft.y + nodeBottomRight.y) / 2;

        switch (tid) {
            case 0:
                topLeft.data[childIndex] = { midX, nodeTopLeft.y };
                bottomRight.data[childIndex] = { nodeBottomRight.x, midY };
                if (count[0] > 0) {
                    start.data[childIndex] = nodeStart;
                    end.data[childIndex] = nodeStart + count[0] - 1;
                }
                break;

            case 1:
                topLeft.data[childIndex] = { nodeTopLeft.x, nodeTopLeft.y };
                bottomRight.data[childIndex] = { midX, midY };
                if (count[1] > 0) {
                    start.data[childIndex] = nodeStart + count[0];
                    end.data[childIndex] = start.data[childIndex] + count[1] - 1;
                }
                break;

            case 2:
                topLeft.data[childIndex] = { nodeTopLeft.x, midY };
                bottomRight.data[childIndex] = { midX, nodeBottomRight.y };
                if (count[2] > 0) {
                    start.data[childIndex] = nodeStart + count[0] + count[1];
                    end.data[childIndex] = start.data[childIndex] + count[2] - 1;
                }
                break;

            case 3:
                topLeft.data[childIndex] = { midX, midY };
                bottomRight.data[childIndex] = { nodeBottomRight.x, nodeBottomRight.y };
                if (count[3] > 0) {
                    start.data[childIndex] = nodeStart + count[0] + count[1] + count[2];
                    end.data[childIndex] = nodeEnd;
                }
                break;
        }
    }

    if (tid == 0) {
        isLeaf.data[nodeIndex] = false;
    }
}

template<typename N>
inline float getDistance( N pos1, N pos2 ) {
    N delta = pos1 - pos2;
    N deltaSquared = pow(delta, 2);
    return sqrt(deltaSquared.x + deltaSquared.y);
}

inline bool isCollide(float2 pos1, float radius1, float2 cm, float collisionThreshold ) {
    return (radius1 * 2 + collisionThreshold) > getDistance(pos1, cm);
}

inline bool isCollide(float2 pos1, float radius1, float2 pos2, float radius2, float collisionThreshold ) {
    return (radius1 + radius2 + collisionThreshold) > getDistance(pos1, pos2);
}

inline void computeBarnesHuntForce( constant NodeMemberData<float2>& topLeft,
                                    constant NodeMemberData<float2>& bottomRight,
                                    constant NodeMemberData<float2>& centerOfMass,
                                    constant NodeMemberData<bool>& isLeaf,
                                    float2 bodyPos,
                                    float bodyRadius,
                                    thread float2& bodyAccel,
                                    int nodeIndex,
                                    int bodyIndex,
                                    float width,
                                    constant PhysicsParams &physics
) {
    uint stack[32];
    float widthStack[32];
    int stackIdx = 0;
    
    stack[stackIdx] = nodeIndex;
    widthStack[stackIdx] = width;
    stackIdx++;
        
    while (stackIdx > 0) {
        stackIdx--;
        uint curIndex = stack[stackIdx];
        float curWidth = widthStack[stackIdx];
        
        if (curIndex >= topLeft.size || curIndex < 0.0h) {
            continue;
        }
        
        float2 curCOM = centerOfMass.data[curIndex];
        bool curIsLeaf = isLeaf.data[curIndex];

        if (curIsLeaf) {
             if (curCOM.x != -1.0h) {
                 float2 delta = curCOM - bodyPos;
                 float distanceSquared = dot(delta, delta) + physics.epsilon;
                 float distance = sqrt(distanceSquared);
                 
                 if (distance > physics.epsilon) {
                     float2 direction = delta / distance;
                     float repulsionMagnitude = physics.edgeRepulsion / distanceSquared;
                     bodyAccel -= direction * repulsionMagnitude;
                 }
             }
             continue;
         }

        float sd = curWidth / getDistance(bodyPos, curCOM);
        if (sd < physics.theta) {
            if (!isCollide(bodyPos, bodyRadius, curCOM, physics.collisionThreshold)) {
                float2 delta = curCOM - bodyPos;
                float distanceSquared = dot(delta, delta) + physics.epsilon;
                float distance = sqrt(distanceSquared);
                
                if (distance > physics.epsilon) {
                    float2 direction = delta / distance;
                    float repulsionMagnitude = physics.edgeRepulsion / distanceSquared;
                    bodyAccel -= direction * repulsionMagnitude;
                }
            }
            continue;
        }

        float halfWidth = curWidth / 2.0f;
        if (stackIdx + 4 <= 32) {
            stack[stackIdx] = (curIndex * 4) + 4;
            widthStack[stackIdx] = halfWidth;
            stackIdx++;
            
            stack[stackIdx] = (curIndex * 4) + 3;
            widthStack[stackIdx] = halfWidth;
            stackIdx++;
            
            stack[stackIdx] = (curIndex * 4) + 2;
            widthStack[stackIdx] = halfWidth;
            stackIdx++;
            
            stack[stackIdx] = (curIndex * 4) + 1;
            widthStack[stackIdx] = halfWidth;
            stackIdx++;
        }
    }
}

inline void computeDirectSumForce(device BodyMemberData<float2>& position,
                                  float2 bodyPos,
                                  thread float2& bodyAccel,
                                  constant PhysicsParams &physics,
                                  uint bid,
                                  ushort tid,
                                  ushort simd_lane_id,
                                  ushort threads_per_threadgroup,
                                  ushort threads_per_simdgroup
) {
    uint gid = bid * threads_per_threadgroup + tid;
    float2 force = { 0.0f, 0.0f };
    
    for (int base = 0; base < (int)numBodies; base += threads_per_simdgroup) {
        uint j = base + simd_lane_id;
        
        float2 otherPos;
        if (j < numBodies) {
            otherPos = position.data[j];
        } else {
            otherPos = { 0.0f, 0.0f };
        }
        
        for (short lane = 0; lane < threads_per_simdgroup && (base + lane) < (int)numBodies; ++lane) {
            float2 pos = simd_broadcast(otherPos, lane);
            
            uint otherIdx = base + lane;
            if (otherIdx == gid)
                continue;
            
            float2 delta = pos - bodyPos;
            float distanceSquared = dot(delta, delta) + physics.epsilon;
            float distance = sqrt(distanceSquared);
            
            if (distance > physics.epsilon) {
                float2 direction = delta / distance;
                float repulsionMagnitude = physics.edgeRepulsion / distanceSquared;
                force -= direction * repulsionMagnitude;
            }
        }
    }
    
    bodyAccel += force;
}

inline uint floatToUInt(float f) {
    uint u = as_type<uint>(f);
    uint mask = (u >> 31) ? 0xFFFFFFFF : 0x80000000;
    return u ^ mask;
}

inline float uintToFloat(uint u) {
    uint mask = (u >> 31) ? 0x80000000 : 0xFFFFFFFF;
    return as_type<float>(u ^ mask);
}

inline void computeEdgesForce( const thread float2& bodyPos,
                               thread float2& bodyAccel,
                              
                               device BodyMemberData<float2>& position,
                               constant BodyMemberData<uint>& offsets,
                              
                               constant EdgeMemberData<uint>& edgesIndiciesData,
                               device EdgeMemberData<uint>& edgeAnglesData,
                              
                               constant PhysicsParams &physics,
                               uint bid,
                               ushort tid,
                               ushort simd_lane_id,
                               ushort threads_per_threadgroup,
                               ushort threads_per_simdgroup
) {
    uint gid = bid * threads_per_threadgroup + tid;
    
    uint startIdx = offsets.data[gid];
    uint endIdx = offsets.data[gid + 1];
    uint numConns = endIdx - startIdx;
    
    float2 force = { 0.0f, 0.0f };
        
    for (int base = 0; base < (int)numConns; base += threads_per_simdgroup) {
        uint baseIdx = base + simd_lane_id;
        
        float2 otherPos;
        if (baseIdx < numConns) {
            uint otherIdx = edgesIndiciesData.data[startIdx + baseIdx];
            otherPos = position.data[otherIdx];
        } else {
            otherPos = { 0.0f, 0.0f };
        }
        
        simdgroup_barrier(mem_flags::mem_threadgroup);
        
        // compute forces
        for (int lane = 0; lane < threads_per_simdgroup; ++lane) {
            uint edgeIdx = base + lane;
            uint insertIdx = edgeIdx + startIdx;
            
            if (edgeIdx >= numConns)
                continue;

            float2 pos = simd_broadcast(otherPos, lane);
            
            float2 delta = pos - bodyPos;
            float distanceSquared = dot(delta, delta) + physics.epsilon;
            float distance = sqrt(distanceSquared);
            

            if (edgeIdx < numConns) {
                float angle = atan2(delta.y, delta.x);
                float normalizedAngle = (angle + M_PI_F) / (2.0f * M_PI_F);
                
                edgeAnglesData.data[insertIdx] = floatToUInt(angle);
            } else {
                edgeAnglesData.data[insertIdx] = 0xFFFFFFFF;
            }
            
            if (distance > physics.epsilon && computeEdges) {
                float2 direction = delta / distance;
                float restLength = physics.edgeAttraction;
                float displacement = distance - restLength;
                float springMagnitude = physics.springConstant * displacement;
                force += direction * springMagnitude;
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    bodyAccel += force;
}

kernel void computeForceKernel(constant NodeMemberData<float2>& topLeft [[buffer(NODE_TOP_LEFT_IDX)]],
                               constant NodeMemberData<float2>& bottomRight [[buffer(NODE_BOTTOM_RIGHT_IDX)]],
                               constant NodeMemberData<float2>& centerOfMass [[buffer(NODE_CENTER_OF_MASS_IDX)]],
                               constant NodeMemberData<bool>& isLeaf [[buffer(NODE_IS_LEAF_IDX)]],
                               //Body
                               device BodyMemberData<float>& mass [[buffer(BODY_MASS_IDX)]],
                               constant BodyMemberData<float>& radius [[buffer(BODY_RADIUS_IDX)]],
                               device BodyMemberData<float2>& position [[buffer(BODY_POSITION_IDX)]],
                               device BodyMemberData<float2>& velocity [[buffer(BODY_VELOCITY_IDX)]],
                               device BodyMemberData<float2>& acceleration [[buffer(BODY_ACCELERATION_IDX)]],
                               device BodyMemberData<uint>& initialIdx [[buffer(BODY_INITIAL_IDX_IDX)]],
                               constant BodyMemberData<uint>& offsets [[buffer(BODY_EDGE_OFFSETS_IDX)]],
                               //Connections
                               constant EdgeMemberData<uint>& edgeTerminationsData [[buffer(EDGE_TERMINATIONS_IDX)]],
                               device EdgeMemberData<uint>& edgeAnglesData [[buffer(EDGE_ANGLES_IDX)]],
                               
                               constant PhysicsParams &physics [[buffer(PHYSICS_PARAMS_IDX)]],
                               ushort tid [[thread_position_in_threadgroup]],
                               uint bid [[threadgroup_position_in_grid]],
                               ushort simd_lane_id [[thread_index_in_simdgroup]],
                               ushort threads_per_threadgroup [[threads_per_threadgroup]],
                               ushort threads_per_simdgroup [[threads_per_simdgroup]]
) {
    uint gid = bid * threads_per_threadgroup + tid;
    
    if (gid >= numBodies)
        return;

    uint bodyInitialIdx = initialIdx.data[gid];
    float2 bodyPos = position.data[gid];
    float bodyMass = mass.data[gid];

    float bodyRadius = radius.data[bodyInitialIdx];
    float2 bodyVel = velocity.data[bodyInitialIdx];
    float2 bodyAccel = { 0.0f, 0.0f };

    if (useBarnes) {
        float width = bottomRight.data[0].x - topLeft.data[0].x;
        computeBarnesHuntForce(topLeft, bottomRight, centerOfMass, isLeaf, bodyPos, bodyRadius, bodyAccel, 0, gid, width, physics);
    } else {
        computeDirectSumForce(position, bodyPos, bodyAccel, physics, bid, tid, simd_lane_id, threads_per_threadgroup, threads_per_simdgroup);
    }
    
    if (gid < edgeTerminationsData.size) {
        computeEdgesForce(bodyPos, bodyAccel, position, offsets, edgeTerminationsData, edgeAnglesData, physics, bid, tid, simd_lane_id, threads_per_threadgroup, threads_per_simdgroup);
    }
    
    bodyVel *= physics.damping;
    bodyVel += bodyAccel * physics.dt;
    
    float2 velocitySquared = pow(bodyVel, 2);
    float velMagnitude = sqrt(velocitySquared.x + velocitySquared.y);
    
    const float MAX_VELOCITY = 2.0f;
    if (velMagnitude > MAX_VELOCITY) {

    }
    
    bodyPos += bodyVel * physics.dt;
    
    position.data[bodyInitialIdx] = bodyPos;
    velocity.data[bodyInitialIdx] = bodyVel;
    acceleration.data[bodyInitialIdx] = bodyAccel;

    mass.data[bodyInitialIdx] = bodyMass;
    initialIdx.data[bodyInitialIdx] = bodyInitialIdx;
}

constant int ITEMS = 32;

kernel void sortEdgeAngles( device EdgeMemberData<uint>& edgeTerminationsSortedData [[buffer(EDGE_TERMINATIONS_SORTED_IDX)]],
                            constant EdgeMemberData<uint>& edgeAnglesData [[buffer(EDGE_ANGLES_IDX)]],
                            constant EdgeMemberData<uint>& edgeTerminationsData [[buffer(EDGE_TERMINATIONS_IDX)]],
                            constant BodyMemberData<uint>& bodyOffsets [[buffer(BODY_EDGE_OFFSETS_IDX)]],
                            uint gid [[threadgroup_position_in_grid]],
                            ushort lane_id [[thread_index_in_threadgroup]],
                            ushort simd_size [[threads_per_simdgroup]]
) {
    if (gid >= numBodies)
        return;
    
    uint startIdx = bodyOffsets.data[gid];
    uint endIdx = bodyOffsets.data[gid + 1];
    uint numConns = endIdx - startIdx;
    
    if (numConns == 0)
        return;
    
    uint localKeys[ITEMS];
    uint localValues[ITEMS];
    
    threadgroup uint scratchKeys[MAX_EDGES];
    threadgroup uint scratchValues[MAX_EDGES];

    for (int i = 0; i < ITEMS; i++) {
        uint localOffset = lane_id + i * simd_size;
        
        if (localOffset < numConns) {
            uint globalIdx = startIdx + localOffset;
            localKeys[i] = edgeAnglesData.data[globalIdx];
            localValues[i] = edgeTerminationsData.data[globalIdx];
        } else {
            localKeys[i] = 0xFFFFFFFF;
            localValues[i] = 0;
        }
    }
    
    for (int bit = 0; bit < 32; bit++) {
        
        uint totalZeros = 0;
        for (uint i = 0; i < ITEMS; i++) {
            bool isZero = !((localKeys[i] >> bit) & 1);
            totalZeros += simd_sum(isZero ? 1 : 0);
        }
        
        uint targetIndices[ITEMS];
        uint runningZeros = 0;
        
        for (uint i = 0; i < ITEMS; i++) {
            bool isZero = !((localKeys[i] >> bit) & 1);
            uint localRank = simd_prefix_exclusive_sum(isZero ? 1 : 0);
            uint rowZeros = simd_broadcast(localRank + (isZero ? 1 : 0), simd_size - 1);
            uint globalIdx = lane_id + i * simd_size;
            
            uint targetIdx;
            if (isZero) {
                targetIdx = runningZeros + localRank;
            } else {
                uint onesBefore = globalIdx - (runningZeros + localRank);
                targetIdx = totalZeros + onesBefore;
            }
            
            targetIndices[i] = targetIdx;
            runningZeros += rowZeros;
        }
        
        for (uint i = 0; i < ITEMS; i++) {
            scratchKeys[targetIndices[i]] = localKeys[i];
            scratchValues[targetIndices[i]] = localValues[i];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (uint i = 0; i < ITEMS; i++) {
            uint idx = lane_id + i * simd_size;
            localKeys[i] = scratchKeys[idx];
            localValues[i] = scratchValues[idx];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    for (int i = 0; i < ITEMS; i++) {
        uint localOffset = lane_id + i * simd_size;
        
        if (localOffset < numConns) {
            uint globalIdx = startIdx + localOffset;
            edgeTerminationsSortedData.data[globalIdx] = localValues[i];
        }
    }
}

#endif
