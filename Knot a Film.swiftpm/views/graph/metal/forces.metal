//
//  forces.metal
//  Knot a Film
//
//  Created by Owen O'Malley on 12/2/25.
//

// based on https://github.com/Hsin-Hung/N-body-simulation
//
//  forces.metal
//  Knot a Film
//

#include <metal_stdlib>
#include "SharedWithMetal.h"

using namespace metal;

constant uint   numBodies          [[ function_constant(FC_NUM_BODIES) ]];
constant uint   numConnections     [[ function_constant(FC_NUM_CONNECTIONS) ]];
constant uint   numNodes           [[ function_constant(FC_NUM_NODES) ]];
constant uint   leafLimit          [[ function_constant(FC_LEAF_LIMIT) ]];

constant int   blockSize          [[ function_constant(FC_BLOCK_SIZE) ]];
constant bool  useBarnes          [[ function_constant(FC_USE_BARNES) ]];
constant bool  computeConnections [[ function_constant(FC_COMPUTE_CONNECTIONS) ]];

constant float springConstant     [[ function_constant(FC_SPRING_CONSTANT) ]];
constant float edgeRepulsion      [[ function_constant(FC_EDGE_REPULSION) ]];
constant float edgeAttraction     [[ function_constant(FC_EDGE_ATTRACTION) ]];
constant float epsilon            [[ function_constant(FC_EPSILON) ]];
constant float dt                 [[ function_constant(FC_DT) ]];
constant float theta              [[ function_constant(FC_THETA) ]];
constant float collisionThreshold [[ function_constant(FC_COLLISION_THRESHOLD) ]];
constant float damping            [[ function_constant(FC_DAMPING) ]];

kernel void initalizeBodies ( device BodyData& bodyData [[buffer(0)]] ,
                              constant PhysicsParams& params [[buffer(1)]],
                              uint gid [[thread_position_in_grid]])
{
    if (gid >= numBodies) {
        return;
    }

    float maxDistance = 0.8f;
    float minDistance = 0.3f;
    float2 centerPos = { 0.0h, 0.0h };

    float angle = 2.0f * M_PI_F * (float(gid) / (float)(numBodies - 1));
    float radius = (maxDistance - minDistance) * (fract(sin(float(gid) * 12.9898f) * 43758.5453f)) + minDistance;

    float x = centerPos.x + radius * cos(angle);
    float y = centerPos.y + radius * sin(angle);

    Body body;
    body.mass = 1.0f;
    body.radius = 0.001f;
    body.position = { x, y };
    body.velocity = { 0.0f, 0.0f };
    body.acceleration = { 0.0f, 0.0f };
    body.initialIdx = gid;
    bodyData.bodies[gid] = body;
}

/*
----------------------------------------------------------------------------------------
RESET KERNEL
----------------------------------------------------------------------------------------
*/

kernel void resetKernel(device NodeData &nodeData [[buffer(0)]],
                        device atomic_int *mutex [[buffer(1)]],
                        constant BodyData &bodyData [[buffer(2)]],
                        uint gid [[thread_position_in_grid]])
{
    if (gid < uint(nodeData.numNodes)) {
        nodeData.nodes[gid].topLeft = {INFINITY, -INFINITY};
        nodeData.nodes[gid].bottomRight = {-INFINITY, INFINITY};
        nodeData.nodes[gid].centerOfMass = {-1.0h, -1.0h};
        nodeData.nodes[gid].totalMass = 0.0;
        nodeData.nodes[gid].isLeaf = true;
        nodeData.nodes[gid].start = -1;
        nodeData.nodes[gid].end = -1;
        atomic_store_explicit(&mutex[gid], 0, memory_order_relaxed);
    }

    if (gid == 0) {
        nodeData.nodes[0].start = 0;
        nodeData.nodes[0].end = numBodies - 1;
    }
}

/*
----------------------------------------------------------------------------------------
COMPUTE BOUNDING BOX
----------------------------------------------------------------------------------------
*/

kernel void computeBoundingBoxKernel(device NodeData &nodeData [[buffer(0)]],
                                     constant BodyData &bodyData [[buffer(1)]],
                                     device atomic_int *mutex [[buffer(2)]],
                                     
                                     uint gid [[thread_position_in_grid]],
                                     ushort simd_lane_id [[thread_index_in_simdgroup]],
                                     ushort simd_group_id [[simdgroup_index_in_threadgroup]])
{
    float topLeftX = INFINITY;
    float topLeftY = -INFINITY;
    float bottomRightX = -INFINITY;
    float bottomRightY = INFINITY;

    if (gid < (uint)numBodies) {
        Body body = bodyData.bodies[gid];
        topLeftX = body.position.x;
        topLeftY = body.position.y;
        bottomRightX = body.position.x;
        bottomRightY = body.position.y;
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
        
        nodeData.nodes[0].topLeft.x = -10.0f;//fmin(nodeData.nodes[0].topLeft.x, topLeftX - paddingX);
        nodeData.nodes[0].topLeft.y = 10.0f;//fmax(nodeData.nodes[0].topLeft.y, topLeftY + paddingY);
        nodeData.nodes[0].bottomRight.x = 10.0f;//fmax(nodeData.nodes[0].bottomRight.x, bottomRightX + paddingX);
        nodeData.nodes[0].bottomRight.y = -10.0f;//fmin(nodeData.nodes[0].bottomRight.y, bottomRightY - paddingY);
        
        atomic_store_explicit(&mutex[0], 0, memory_order_relaxed);
    }
}

/*
----------------------------------------------------------------------------------------
CONSTRUCT QUAD TREE
----------------------------------------------------------------------------------------
*/

inline int getQuadrant(float2 topLeft,
                       float2 bottomRight,
                       float x,
                       float y)
{
    float midX = (topLeft.x + bottomRight.x) * 0.5f;
    float midY = (topLeft.y + bottomRight.y) * 0.5f;
    
    bool left  = (midX >= x);
    bool top   = (midY <= y);

    return left
        ? (top ? 2 : 3)
        : (top ? 1 : 4);
}

inline void computeCenterOfMass(thread Node &curNode,
                                constant BodyData &bodyData,
                                int start,
                                int end,
                                uint tid,
                                ushort simd_lane_id,
                                ushort threadgroup_size)
{
    int total = end - start + 1;
    int sz = (total + threadgroup_size - 1) / threadgroup_size;
    int s = tid * sz + start;
    float M = 0.0;
    float2 R = { 0.0f, 0.0f };

    for (int i = s; i < s + sz; ++i) {
        if (i <= end) {
            thread Body body = bodyData.bodies[i];
            M += body.mass;
            R += body.mass * body.position;
        }
    }

    M = simd_sum(M);
    R.x = simd_sum(R.x);
    R.y = simd_sum(R.y);

    
    if (simd_lane_id == 0 && M > 0.0f) { //store result from first lane of each simdgroup
        R /= M;
        curNode.totalMass = M;
        curNode.centerOfMass = R;
    }
}

inline void countBodies(constant BodyData &bodyData,
                        float2 topLeft,
                        float2 bottomRight,
                        threadgroup int* __restrict__ count,
                        int start,
                        int end,
                        uint tid,
                        ushort threadgroup_size)
{
    if (tid < 4)
        count[tid] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = start + tid; i <= end; i += threadgroup_size) {
        Body body = bodyData.bodies[i];
        int q = getQuadrant(topLeft, bottomRight, body.position.x, body.position.y);
        atomic_fetch_add_explicit((threadgroup atomic_int*)&count[q - 1], 1, memory_order_relaxed);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void computeOffset(threadgroup int* __restrict__ count,
                          int start,
                          uint tid)
{
    if (tid < 4) {
        int offset = start;
        for (int i = 0; i < int(tid); ++i) {
            offset += count[i];
        }
        count[tid + 4] = offset;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void groupBodies(constant BodyData &bodyData,
                        device BodyData &bufferData,
                        float2 topLeft,
                        float2 bottomRight,
                        threadgroup int* __restrict__ count,
                        int start,
                        int end,
                        uint tid,
                        ushort threadgroup_size)
{
    threadgroup int* count2 = &count[4];
    for (int i = start + tid; i <= end; i += threadgroup_size)
    {
        Body body = bodyData.bodies[i];
        int q = getQuadrant(topLeft, bottomRight, body.position.x, body.position.y);
        int dest = atomic_fetch_add_explicit((threadgroup atomic_int*)&count2[q - 1], 1, memory_order_relaxed);
        bufferData.bodies[dest] = body;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void constructQuadTreeKernel(device NodeData &nodeData [[buffer(0)]],
                                    constant BodyData &bodyData [[buffer(1)]],
                                    device BodyData &bufferData [[buffer(2)]],
                                    constant int &nodeOffset [[buffer(3)]],
                                    
                                    threadgroup int* __restrict__ count [[threadgroup(0)]],
                                    
                                    ushort tid [[thread_position_in_threadgroup]],
                                    uint bid [[threadgroup_position_in_grid]],
                                    ushort simd_lane_id [[thread_index_in_simdgroup]],
                                    ushort threadgroup_size [[threads_per_threadgroup]])
{
    uint nodeIndex = nodeOffset + bid;

    if (nodeIndex >= nodeData.numNodes)
        return;

    thread Node curNode = nodeData.nodes[nodeIndex];
    int start = curNode.start, end = curNode.end;
    float2 topLeft = curNode.topLeft, bottomRight = curNode.bottomRight;

    if (start == -1 || end == -1)
        return;

    computeCenterOfMass(curNode, bodyData, start, end, tid, simd_lane_id, threadgroup_size);
    
    if (tid == 0) {
        nodeData.nodes[nodeIndex] = curNode;
    }
    
    //threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (nodeIndex >= uint(leafLimit) || start == end) {
        for (int i = start; i <= end; ++i) {
            bufferData.bodies[i] = bodyData.bodies[i];
        }
        return;
    }

    countBodies(bodyData, topLeft, bottomRight, count, start, end, tid, threadgroup_size);
    computeOffset(count, start, tid);
    groupBodies(bodyData, bufferData, topLeft, bottomRight, count, start, end, tid, threadgroup_size);


    if (tid < 4) { // update child bound
        uint quadIndex = tid + 1;
        Node node = nodeData.nodes[(nodeIndex * 4) + quadIndex];

        float midX = (topLeft.x + bottomRight.x) / 2;
        float midY = (topLeft.y + bottomRight.y) / 2;

        switch (tid) {
            case 0: // quadrant 1
                node.topLeft     = { midX, topLeft.y };
                node.bottomRight = { bottomRight.x, midY };
                if (count[0] > 0) {
                    node.start = start;
                    node.end   = start + count[0] - 1;
                }
                break;

            case 1: // quadrant 2
                node.topLeft     = { topLeft.x, topLeft.y };
                node.bottomRight = { midX, midY };
                if (count[1] > 0) {
                    node.start = start + count[0];
                    node.end   = node.start + count[1] - 1;
                }
                break;

            case 2: // quadrant 3
                node.topLeft     = { topLeft.x, midY };
                node.bottomRight = { midX, bottomRight.y };
                if (count[2] > 0) {
                    node.start = start + count[0] + count[1];
                    node.end   = node.start + count[2] - 1;
                }
                break;

            case 3: // quadrant 4
                node.topLeft     = { midX, midY };
                node.bottomRight = { bottomRight.x, bottomRight.y };
                if (count[3] > 0) {
                    node.start = start + count[0] + count[1] + count[2];
                    node.end   = end;
                }
                break;
        }

        nodeData.nodes[(nodeIndex * 4) + quadIndex] = node;
    }

    if (tid == 0) {
        curNode.isLeaf = false;
        nodeData.nodes[nodeIndex] = curNode;
    }

}

/*
----------------------------------------------------------------------------------------
COMPUTE FORCE
----------------------------------------------------------------------------------------
*/

template<typename N>
inline float getDistance( N pos1, N pos2 ) {
    N delta = pos1 - pos2;
    N deltaSquared = pow(delta, 2);
    return sqrt(deltaSquared.x + deltaSquared.y);
}

inline bool isCollide(Body b1, float2 cm, float collisionThreshold ) {
    return (b1.radius * 2 + collisionThreshold) > getDistance(b1.position, cm);
}

inline bool isCollide( thread const Body &b1, thread const Body &b2, float collisionThreshold ) {
    return (b1.radius + b2.radius + collisionThreshold) > getDistance(b1.position, b2.position);
}

inline void computeBarnesHuntForce( constant const NodeData &nodeData,
                                    const device BodyData &bodyData,
                                    thread Body& bi,
                                    int nodeIndex,
                                    int bodyIndex,
                                    float width,
                                    constant PhysicsParams &physics )
{
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
        
        if (curIndex >= nodeData.numNodes || curIndex < 0.0h) {
            continue;
        }
        
        Node curNode = nodeData.nodes[curIndex];

        if (curNode.isLeaf) {
             if (curNode.centerOfMass.x != -1.0h) {
                 float2 delta = curNode.centerOfMass - bi.position;
                 float distanceSquared = dot(delta, delta) + physics.epsilon;
                 float distance = sqrt(distanceSquared);
                 
                 if (distance > physics.epsilon) {
                     float2 direction = delta / distance;
                     float repulsionMagnitude = physics.edgeRepulsion / distanceSquared;
                     bi.acceleration -= direction * repulsionMagnitude;
                 }
             }
             continue;
         }

        float sd = curWidth / getDistance(bi.position, curNode.centerOfMass);
        if (sd < physics.theta) {
            if (!isCollide(bi, curNode.centerOfMass, physics.collisionThreshold)) {
                float2 delta = curNode.centerOfMass - bi.position;
                float distanceSquared = dot(delta, delta) + physics.epsilon;
                float distance = sqrt(distanceSquared);
                
                if (distance > physics.epsilon) {
                    float2 direction = delta / distance;
                    float repulsionMagnitude = physics.edgeRepulsion / distanceSquared;
                    bi.acceleration -= direction * repulsionMagnitude;
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

inline void computeDirectSumForce( const device BodyData& bodyData,
                                   thread Body& bi,
                                   constant PhysicsParams &physics,
                                   uint gid,
                                   ushort simd_lane_id )
{
    if (gid >= numBodies) {
        return;
    }
    
    float2 force = { 0.0f, 0.0f };
    
    for (uint base = 0; base < numBodies; base += 32) {
        uint j = base + simd_lane_id;
        
        thread Body bj;
        if (j < numBodies) {
            bj = bodyData.bodies[j];
        } else {
            bj.position = { 0.0f, 0.0f };
            bj.mass = 0.0f;
            bj.radius = 0.0f;
        }
        
        for (uint lane = 0; lane < 32 && (base + lane) < numBodies; ++lane) {
            float2 pos = simd_broadcast(bj.position, lane);
            //float mass = simd_broadcast(bj.mass, lane);
            //float radius = simd_broadcast(bj.radius, lane);
            
            uint otherIdx = base + lane;
            if (otherIdx == gid)
                continue; // Skip self
            
            float2 delta = pos - bi.position;
            float distanceSquared = dot(delta, delta) + physics.epsilon;
            float distance = sqrt(distanceSquared);
            
            if (distance > physics.epsilon) {
                float2 direction = delta / distance;
                float repulsionMagnitude = physics.edgeRepulsion / distanceSquared;
                force -= direction * repulsionMagnitude;
            }
        }
    }
    
    bi.acceleration += force;
}

inline void computeConnectionsForce( const device BodyData& bodyData,
                                     thread Body& bi,
                                     thread PerBodyConnectionsData& bConnections,
                                     constant uint& numBodiesWithConnections,
                                     constant PhysicsParams &physics,
                                     uint gid,
                                     ushort simd_lane_id )
{
    if (gid >= numBodies) {
        return;
    }
    
    float2 force = { 0.0f, 0.0f };
    
    for (uint base = 0; base < bConnections.numPerBodyConnections; base += 32) {
        uint localIdx = base + simd_lane_id;
        
        Body bj;
        if (localIdx < bConnections.numPerBodyConnections) {
            uint connectionIdx = bConnections.perBodyConnections[localIdx];
            bj = bodyData.bodies[connectionIdx];
        } else {
            bj.position = { 0.0f, 0.0f };
            bj.mass = 0.0f;
        }
        
        for (uint lane = 0; lane < 32 && (base + lane) < bConnections.numPerBodyConnections; ++lane) {
            float2 pos = simd_broadcast(bj.position, lane);
            
            float2 delta = pos - bi.position;
            float distanceSquared = dot(delta, delta) + physics.epsilon;
            float distance = sqrt(distanceSquared);
            
            if (distance > physics.epsilon) {
                float2 direction = delta / distance;
                float restLength = physics.edgeAttraction;
                float displacement = distance - restLength;
                float springMagnitude = physics.springConstant * displacement;
                force += direction * springMagnitude;
            }
        }
    }
    
    bi.acceleration += force;
}

kernel void computeForceKernel( constant const NodeData &nodeData [[buffer(0)]],
                                constant const ConnectionsData& connectionsData [[buffer(1)]],
                                device BodyData &bodyData [[buffer(2)]],
                                constant const PhysicsParams &physics [[buffer(3)]],

                                uint gid [[thread_position_in_grid]],
                                ushort simd_lane_id [[thread_index_in_simdgroup]] )
{
    if (gid >= numBodies) {
        return;
    }
    
    thread Body bi = bodyData.bodies[gid];
    bi.acceleration = { 0.0f, 0.0f };

    if (useBarnes) {
        float width = nodeData.nodes[0].bottomRight.x - nodeData.nodes[0].topLeft.x;
        computeBarnesHuntForce(nodeData, bodyData, bi, 0, gid, width, physics);
    } else {
        computeDirectSumForce(bodyData, bi, physics, gid, simd_lane_id);
    }
    
    if (computeConnections && gid < connectionsData.numConnections) {
        thread PerBodyConnectionsData bConnections = connectionsData.connections[gid];
        computeConnectionsForce(bodyData, bi, bConnections, connectionsData.numConnections, physics, gid, simd_lane_id);
    }
    
    bi.velocity *= physics.damping;
    bi.velocity += bi.acceleration * physics.dt;
    
    float2 velocitySquared = pow(bi.velocity, 2);
    float velMagnitude = sqrt(velocitySquared.x + velocitySquared.y);
    
    const float MAX_VELOCITY = 2.0f;
    if (velMagnitude > MAX_VELOCITY) {
        //bi.velocity *= (MAX_VELOCITY / velMagnitude);
    }
    
    bi.position += bi.velocity * physics.dt;
    bodyData.bodies[bi.initialIdx] = bi;
}

/*
----------------------------------------------------------------------------------------
EDGE
----------------------------------------------------------------------------------------
*/
kernel void coalesceConnectionsIndices( device const uint2* __restrict__ connections  [[buffer(0)]],
                                        device ConnectionsData& connectionsData [[buffer(1)]],
                                        constant uint& numConnections  [[buffer(2)]],
                                        constant uint& numBodies  [[buffer(3)]],
                                        threadgroup uint2* __restrict__ connectionsTile [[threadgroup(0)]],
                                        uint gid  [[thread_position_in_grid]],
                                        uint tid  [[thread_index_in_threadgroup]],
                                        ushort threadgroup_size  [[threads_per_threadgroup]])
{
    if (gid >= numBodies)
        return;
    
    thread PerBodyConnectionsData perBodyConnectionsData;
    thread uint appendIdx = 0;
    
    for (int i = 0; i < MAX_CONNECTIONS; ++i) {
        perBodyConnectionsData.perBodyConnections[i] = UINT_MAX;
    }

    const int numConnectionTiles = (numConnections + threadgroup_size - 1) / threadgroup_size;

    for (int tile = 0; tile < numConnectionTiles; ++tile) {
        uint idx = tile * threadgroup_size + tid;
        if (idx < numConnections) {
            connectionsTile[tid] = connections[idx];
        } else {
            connectionsTile[tid] = uint2(UINT_MAX, UINT_MAX);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int b = 0; b < threadgroup_size; ++b) {
            uint j = tile * threadgroup_size + b;
            if (j >= numConnections)
                break;

            uint2 c = connectionsTile[b];

            if ((c.x == gid || c.y == gid) && appendIdx < MAX_CONNECTIONS) {
                uint otherBody = (c.x == gid) ? c.y : c.x;
                perBodyConnectionsData.perBodyConnections[appendIdx] = otherBody;
                appendIdx++;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    perBodyConnectionsData.numPerBodyConnections = appendIdx;
    connectionsData.connections[tid] = perBodyConnectionsData;
}
