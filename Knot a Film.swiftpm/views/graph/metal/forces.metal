//
//  forces.metal
//  Knot a Film
//
//  Created by Owen O'Malley on 12/2/25.
//

// based on https://github.com/Hsin-Hung/N-body-simulation

#include <metal_stdlib>
#include "include/GraphMetalTypes.h"

using namespace metal;

/*
----------------------------------------------------------------------------------------
RESET KERNEL
----------------------------------------------------------------------------------------
*/

kernel void resetKernel(device NodeData &nodeData [[buffer(0)]],
                        device atomic_int *mutex [[buffer(1)]],
                        device BodyData &bodyData [[buffer(2)]],
                        uint gid [[thread_position_in_grid]])
{
    if (gid < uint(nodeData.numNodes)) {
        nodeData.nodes[gid].topLeft = {INFINITY, -INFINITY};
        nodeData.nodes[gid].bottomRight = {-INFINITY, INFINITY};
        nodeData.nodes[gid].centerOfMass = {-1, -1};
        nodeData.nodes[gid].totalMass = 0.0;
        nodeData.nodes[gid].isLeaf = true;
        nodeData.nodes[gid].start = -1;
        nodeData.nodes[gid].end = -1;
        atomic_store_explicit(&mutex[gid], 0, memory_order_relaxed);
    }

    if (gid == 0) {
        nodeData.nodes[0].start = 0;
        nodeData.nodes[0].end = bodyData.numBodies - 1;
    }
}

/*
----------------------------------------------------------------------------------------
COMPUTE BOUNDING BOX
----------------------------------------------------------------------------------------
*/

kernel void computeBoundingBoxKernel(device NodeData &nodeData [[buffer(0)]],
                                     device BodyData &bodyData [[buffer(1)]],
                                     device atomic_int *mutex [[buffer(2)]],
                                     threadgroup half* __restrict__ topLeftX [[threadgroup(0)]],
                                     threadgroup half* __restrict__ topLeftY [[threadgroup(1)]],
                                     threadgroup half* __restrict__ bottomRightX [[threadgroup(2)]],
                                     threadgroup half* __restrict__ bottomRightY [[threadgroup(3)]],
                                     uint tid [[thread_position_in_threadgroup]],
                                     uint gid [[thread_position_in_grid]],
                                     uint threadgroup_size [[threads_per_threadgroup]])
{
    topLeftX[tid] = INFINITY;
    topLeftY[tid] = -INFINITY;
    bottomRightX[tid] = -INFINITY;
    bottomRightY[tid] = INFINITY;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid < uint(bodyData.numBodies)) {
        Body body = bodyData.bodies[gid];
        topLeftX[tid] = body.position.x;
        topLeftY[tid] = body.position.y;
        bottomRightX[tid] = body.position.x;
        bottomRightY[tid] = body.position.y;
    }

    for (uint s = threadgroup_size / 2; s > 0; s >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < s)
        {
            topLeftX[tid] = fmin(topLeftX[tid], topLeftX[tid + s]);
            topLeftY[tid] = fmax(topLeftY[tid], topLeftY[tid + s]);
            bottomRightX[tid] = fmax(bottomRightX[tid], bottomRightX[tid + s]);
            bottomRightY[tid] = fmin(bottomRightY[tid], bottomRightY[tid + s]);
        }
    }

    if (tid == 0) {
        int expected = 0;
        while (!atomic_compare_exchange_weak_explicit(&mutex[0], &expected, 1, 
                                                       memory_order_relaxed, 
                                                       memory_order_relaxed))
        {
            expected = 0;
        }
        

        //bounding of the quad tree construction with some padding that is super gittery
        float rangeX = topLeftX[0] - bottomRightX[0];
        float rangeY = topLeftY[0] - bottomRightY[0];
        float paddingX = max(abs(rangeX) * 0.01f, 0.1f);
        float paddingY = max(abs(rangeY) * 0.01f, 0.1f);
        
        nodeData.nodes[0].topLeft.x = fmin(nodeData.nodes[0].topLeft.x, topLeftX[0] - paddingX);
        nodeData.nodes[0].topLeft.y = fmax(nodeData.nodes[0].topLeft.y, topLeftY[0] + paddingY);
        nodeData.nodes[0].bottomRight.x = fmax(nodeData.nodes[0].bottomRight.x, bottomRightX[0] + paddingX);
        nodeData.nodes[0].bottomRight.y = fmin(nodeData.nodes[0].bottomRight.y, bottomRightY[0] - paddingY);
        
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
    if ((topLeft.x + bottomRight.x) / 2 >= x) { 
        if ((topLeft.y + bottomRight.y) / 2 <= y) {
            return 2;// topLeftTree
        } else {
            return 3;// botLeftTree
        }
    } else {
        if ((topLeft.y + bottomRight.y) / 2 <= y)
        {
            return 1;// topRightTree
        } else {
            return 4;// bottomRightTree
        }
    }
}

inline void updateChildBound(float2 tl, float2 br,
                             thread Node &childNode,
                             int quadrant)
{
    if (quadrant == 1) {
        childNode.topLeft = {(tl.x + br.x) / 2, tl.y};
        childNode.bottomRight = {br.x, (tl.y + br.y) / 2};
    } else if (quadrant == 2) {
        childNode.topLeft = {tl.x, tl.y};
        childNode.bottomRight = {(tl.x + br.x) / 2, (tl.y + br.y) / 2};
    } else if (quadrant == 3) {
        childNode.topLeft = {tl.x, (tl.y + br.y) / 2};
        childNode.bottomRight = {(tl.x + br.x) / 2, br.y};
    } else {
        childNode.topLeft = {(tl.x + br.x) / 2, (tl.y + br.y) / 2};
        childNode.bottomRight = {br.x, br.y};
    }
}

inline void warpReduce(threadgroup volatile float* __restrict__ totalMass,
                       threadgroup volatile float2* __restrict__ centerOfMass,
                       uint tid)
{
    totalMass[tid] += totalMass[tid + 32];
    centerOfMass[tid] += centerOfMass[tid + 32];
    totalMass[tid] += totalMass[tid + 16];
    centerOfMass[tid] += centerOfMass[tid + 16];
    totalMass[tid] += totalMass[tid + 8];
    centerOfMass[tid] += centerOfMass[tid + 8];
    totalMass[tid] += totalMass[tid + 4];
    centerOfMass[tid] += centerOfMass[tid + 4];
    totalMass[tid] += totalMass[tid + 2];
    centerOfMass[tid] += centerOfMass[tid + 2];
    totalMass[tid] += totalMass[tid + 1];
    centerOfMass[tid] += centerOfMass[tid + 1];
}

inline void computeCenterOfMass(thread Node &curNode,
                                device BodyData &bodyData,
                                threadgroup float* __restrict__ totalMass,
                                threadgroup float2* __restrict__ centerOfMass,
                                int start,
                                int end,
                                uint tid,
                                uint threadgroup_size)
{
    int total = end - start + 1;
    int sz = (total + threadgroup_size - 1) / threadgroup_size;
    int s = tid * sz + start;
    float M = 0.0;
    float2 R = { 0.0, 0.0 };

    for (int i = s; i < s + sz; ++i) {
        if (i <= end) {
            Body body = bodyData.bodies[i];
            M += body.mass;
            R += body.mass * body.position;
        }
    }

    totalMass[tid] = M;
    centerOfMass[tid] = R;

    for (uint stride = threadgroup_size / 2; stride > 32; stride >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < stride) {
            totalMass[tid] += totalMass[tid + stride];
            centerOfMass[tid].x += centerOfMass[tid + stride].x;
            centerOfMass[tid].y += centerOfMass[tid + stride].y;
        }
    }

    if (tid < 32) {
        warpReduce(totalMass, centerOfMass, tid);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        centerOfMass[0].x /= totalMass[0];
        centerOfMass[0].y /= totalMass[0];
        curNode.totalMass = totalMass[0];
        curNode.centerOfMass = centerOfMass[0];
    }
}

inline void countBodies(device BodyData &bodyData, 
                        float2 topLeft,
                        float2 bottomRight,
                        threadgroup int* __restrict__ count,
                        int start,
                        int end,
                        uint tid,
                        uint threadgroup_size)
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

inline void groupBodies(device BodyData &bodyData, 
                        device BodyData &bufferData, 
                        float2 topLeft,
                        float2 bottomRight,
                        threadgroup int* __restrict__ count,
                        int start,
                        int end,
                        uint tid,
                        uint threadgroup_size)
{
    threadgroup int *count2 = &count[4];
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
                                    device BodyData &bodyData [[buffer(1)]],
                                    device BodyData &bufferData [[buffer(2)]],
                                    constant int &nodeOffset [[buffer(3)]],
                                    constant int &leafLimit [[buffer(4)]],
                                    threadgroup int* __restrict__ count [[threadgroup(0)]],
                                    threadgroup float* __restrict__ totalMass [[threadgroup(1)]],
                                    threadgroup float2* __restrict__ centerOfMass [[threadgroup(2)]],
                                    uint tid [[thread_position_in_threadgroup]],
                                    uint bid [[threadgroup_position_in_grid]],
                                    uint threadgroup_size [[threads_per_threadgroup]])
{
    uint nodeIndex = nodeOffset + bid;

    if (nodeIndex >= uint(nodeData.numNodes))
        return;

    Node curNode = nodeData.nodes[nodeIndex];
    int start = curNode.start, end = curNode.end;
    float2 topLeft = curNode.topLeft, bottomRight = curNode.bottomRight;

    if (start == -1 || end == -1)
        return;

    computeCenterOfMass(curNode, bodyData, totalMass, centerOfMass, start, end, tid, threadgroup_size);
    
    // Write back the updated node
    if (tid == 0) {
        nodeData.nodes[nodeIndex] = curNode;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (nodeIndex >= uint(leafLimit) || start == end) {
        for (int i = start; i <= end; ++i) {
            bufferData.bodies[i] = bodyData.bodies[i];
        }
        return;
    }

    countBodies(bodyData, topLeft, bottomRight, count, start, end, tid, threadgroup_size);
    computeOffset(count, start, tid);
    groupBodies(bodyData, bufferData, topLeft, bottomRight, count, start, end, tid, threadgroup_size);
    
    if (tid == 0) {
        Node topLNode = nodeData.nodes[(nodeIndex * 4) + 2];
        Node topRNode = nodeData.nodes[(nodeIndex * 4) + 1];
        Node botLNode = nodeData.nodes[(nodeIndex * 4) + 3];
        Node botRNode = nodeData.nodes[(nodeIndex * 4) + 4];

        updateChildBound(topLeft, bottomRight, topLNode, 2);
        updateChildBound(topLeft, bottomRight, topRNode, 1);
        updateChildBound(topLeft, bottomRight, botLNode, 3);
        updateChildBound(topLeft, bottomRight, botRNode, 4);

        curNode.isLeaf = false;

        if (count[0] > 0) {
            topRNode.start = start;
            topRNode.end = start + count[0] - 1;
        }

        if (count[1] > 0) {
            topLNode.start = start + count[0];
            topLNode.end = start + count[0] + count[1] - 1;
        }

        if (count[2] > 0) {
            botLNode.start = start + count[0] + count[1];
            botLNode.end = start + count[0] + count[1] + count[2] - 1;
        }

        if (count[3] > 0) {
            botRNode.start = start + count[0] + count[1] + count[2];
            botRNode.end = end;
        }
        
        // Write back all child nodes
        nodeData.nodes[(nodeIndex * 4) + 1] = topRNode;
        nodeData.nodes[(nodeIndex * 4) + 2] = topLNode;
        nodeData.nodes[(nodeIndex * 4) + 3] = botLNode;
        nodeData.nodes[(nodeIndex * 4) + 4] = botRNode;
        
        nodeData.nodes[nodeIndex] = curNode;
    }
}

/*
----------------------------------------------------------------------------------------
COMPUTE FORCE
----------------------------------------------------------------------------------------
*/

inline float getDistance(float2 pos1, float2 pos2) {
    float2 delta = pos1 - pos2;
    float2 deltaSquared = pow(delta, 2);
    return sqrt(deltaSquared.x + deltaSquared.y);
}


inline bool isCollide(Body b1, float2 cm, float collisionThreshold) {
    return b1.radius * 2 + collisionThreshold > getDistance(b1.position, cm);
}


inline void computeForce( constant const NodeData &nodeData,
                          device BodyData &bodyData,
                          thread Body& bi,
                          int nodeIndex,
                          int bodyIndex,
                          int leafLimit,
                          float width,
                          constant const PhysicsParams &physics)
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

        float2 cm = curNode.centerOfMass;

        if (curNode.isLeaf) {
            if (curNode.centerOfMass.x != -1) {
                float dist = getDistance(bi.position, cm);
                if (dist > 0.01h) {
                    float2 rij = cm - bi.position;
                    float r = sqrt((rij.x * rij.x) + (rij.y * rij.y) + (physics.epsilon * physics.epsilon));
                    float f = (physics.gravity * bi.mass * curNode.totalMass) / (r * r + (physics.epsilon * physics.epsilon));
                    float2 force = { rij.x * f, rij.y * f };

                    bi.acceleration -= (force / bi.mass);
                }
            }
            continue;
        }

        float sd = curWidth / getDistance(bi.position, cm);
        if (sd < physics.theta) {
            float2 rij = cm - bi.position;
            float r = sqrt((rij.x * rij.x) + (rij.y * rij.y) + (physics.epsilon * physics.epsilon));
            float f = (physics.gravity * bi.mass * curNode.totalMass) / (r * r * r + (physics.epsilon * physics.epsilon));
            float2 force = rij * f;

            bi.acceleration -= (force / bi.mass);
            continue;
        }

        float halfWidth = curWidth / 2.0h;
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

kernel void coalesceConnectionsIndices( device const uint2* __restrict__ connections  [[buffer(0)]],
                                        device ConnectionsData& connectionsData  [[buffer(1)]],
                                        constant uint& numConnections  [[buffer(2)]],
                                        constant uint& numBodies  [[buffer(3)]],
                                        threadgroup uint2* __restrict__ connectionsTile [[threadgroup(0)]],
                                        uint tid  [[thread_position_in_grid]],
                                        uint gid  [[thread_index_in_threadgroup]],
                                        uint threadgroup_size  [[threads_per_threadgroup]])
{
    if (tid >= numBodies)
        return;
    
    thread PerBodyConnectionsData perBodyConnectionsData;
    thread uint appendIdx = 0;
    
    for (uint i = 0; i < MAX_CONNECTIONS; ++i) {
        perBodyConnectionsData.perBodyConnections[i] = UINT_MAX;
    }

    const uint numConnectionTiles = (numConnections + threadgroup_size - 1) / threadgroup_size;

    for (uint tile = 0; tile < numConnectionTiles; ++tile) {

        uint idx = tile * threadgroup_size + gid;
        if (idx < numConnections) {
            connectionsTile[gid] = connections[idx];
        } else {
            connectionsTile[gid] = uint2(UINT_MAX, UINT_MAX);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint b = 0; b < threadgroup_size; ++b) {
            uint j = tile * threadgroup_size + b;
            if (j >= numConnections)
                break;

            uint2 c = connectionsTile[b];

            // check both directions
            if ((c.x == tid || c.y == tid) && appendIdx < MAX_CONNECTIONS) {
                uint otherBody = (c.x == tid) ? c.y : c.x;
                perBodyConnectionsData.perBodyConnections[appendIdx] = otherBody;
                appendIdx++;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    perBodyConnectionsData.numConnections = appendIdx;
    connectionsData.connections[tid] = perBodyConnectionsData;
}


inline void computeConnectionsForce( const device BodyData& bodyData,
                                     thread Body& bi,
                                     thread PerBodyConnectionsData& bConnections,
                                     threadgroup Body* __restrict__ tileBlock,
                                     constant uint& numBodiesWithConnections,
                                     constant const PhysicsParams &physics,
                                     uint tid                 [[thread_index_in_threadgroup]],
                                     uint gid                 [[thread_position_in_grid]],
                                     uint threadgroup_size   [[threads_per_threadgroup]])
{

    if (gid >= bodyData.numBodies) {
        return;
    }

    float2 force = { 0.0f, 0.0f };

    const uint numTiles = (bConnections.numConnections + threadgroup_size - 1) / threadgroup_size;

    for (uint tile = 0; tile < numTiles; ++tile) {
        uint idx = tile * threadgroup_size + tid;

        uint connectionIdx = UINT_MAX;
        thread Body body;
        body.position = { 0.0f, 0.0f };
        body.mass = 0.0f;
        body.radius = 0.0f;
        body.isDynamic = false;
        body.velocity = { 0.0f, 0.0f };
        
        if (idx < bConnections.numConnections) {
            connectionIdx = bConnections.perBodyConnections[idx];
            if (connectionIdx != UINT_MAX && connectionIdx < bodyData.numBodies) {
                body = bodyData.bodies[connectionIdx];
            }
        }
        
        tileBlock[tid] = body;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint bodyIdx = 0; bodyIdx < threadgroup_size; ++bodyIdx) {
            uint connectionGlobalIdx = tile * threadgroup_size + bodyIdx;
            if (connectionGlobalIdx >= bConnections.numConnections)
                break;

            Body bodyj = tileBlock[bodyIdx];
            uint otherBodyIdx = bConnections.perBodyConnections[connectionGlobalIdx];
            
            if (otherBodyIdx != UINT_MAX && otherBodyIdx != gid && bodyj.mass > 0.0f) { // Skip invalid connections, self-interactions, and zero-mass bodies
                float2 delta = float2(bodyj.position - bi.position);
                float distSq = (delta.x * delta.x) + (delta.y * delta.y);
                
                if (distSq > 0.0001f) {
                    float r = sqrt(distSq + (physics.epsilon * physics.epsilon));
                    float f = (physics.gravity * bi.mass * bodyj.mass) / (r * r);
                    force += (delta * f) / bi.mass;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    bi.acceleration += 100 * force;
}

kernel void computeForceKernel( constant NodeData &nodeData [[buffer(0)]],
                                constant ConnectionsData &connectionsData [[buffer(1)]],
                                device BodyData &bodyData [[buffer(2)]],
                                constant int &leafLimit [[buffer(3)]],
                                constant PhysicsParams &physics [[buffer(4)]],
                                threadgroup Body* tileBlock [[threadgroup(0)]],

                                uint tid                [[thread_index_in_threadgroup]],
                                uint gid                [[thread_position_in_grid]],
                                uint threadgroup_size   [[threads_per_threadgroup]])
{
    
    float width = nodeData.nodes[0].bottomRight.x - nodeData.nodes[0].topLeft.x;
    
    if (gid < uint(bodyData.numBodies)) {
        thread Body bi = bodyData.bodies[gid];
        thread PerBodyConnectionsData bConnections = connectionsData.connections[gid];
        
        if (bi.isDynamic) {
            bi.acceleration = {0.0, 0.0};
            computeForce(nodeData, bodyData, bi, 0, gid, leafLimit, width, physics);
            computeConnectionsForce(bodyData, bi, bConnections, tileBlock, connectionsData.numBodiesWithConnections, physics, tid, gid, threadgroup_size);
                        
            bi.velocity *= physics.damping;
            bi.velocity += bi.acceleration * physics.dt;
            
            //clamping
            const float MAX_VELOCITY = 100.0f;
            float velMag = sqrt(bi.velocity.x * bi.velocity.x + bi.velocity.y * bi.velocity.y);
            if (velMag > MAX_VELOCITY) {
                bi.velocity *= (MAX_VELOCITY / velMag);
            }
            
            bi.position += bi.velocity * physics.dt;
        }
        bodyData.bodies[gid] = bi;
    }
}

