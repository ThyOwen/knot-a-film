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

kernel void resetKernel(device Node *node [[buffer(0)]],
                        device atomic_int *mutex [[buffer(1)]],
                        constant int &nNodes [[buffer(2)]],
                        constant int &nBodies [[buffer(3)]],
                        uint gid [[thread_position_in_grid]])
{
    if (gid < uint(nNodes)) {
        node[gid].topLeft = {INFINITY, -INFINITY};
        node[gid].bottomRight = {-INFINITY, INFINITY};
        node[gid].centerOfMass = {-1, -1};
        node[gid].totalMass = 0.0;
        node[gid].isLeaf = true;
        node[gid].start = -1;
        node[gid].end = -1;
        atomic_store_explicit(&mutex[gid], 0, memory_order_relaxed);
    }

    if (gid == 0) {
        node[0].start = 0;
        node[0].end = nBodies - 1;
    }
}

/*
----------------------------------------------------------------------------------------
COMPUTE BOUNDING BOX
----------------------------------------------------------------------------------------
*/

kernel void computeBoundingBoxKernel(device Node *node [[buffer(0)]],
                                     device Body *bodies [[buffer(1)]],
                                     device atomic_int *mutex [[buffer(2)]],
                                     constant int &nBodies [[buffer(3)]],
                                     threadgroup float *topLeftX [[threadgroup(0)]],
                                     threadgroup float *topLeftY [[threadgroup(1)]],
                                     threadgroup float *bottomRightX [[threadgroup(2)]],
                                     threadgroup float *bottomRightY [[threadgroup(3)]],
                                     uint tid [[thread_position_in_threadgroup]],
                                     uint gid [[thread_position_in_grid]],
                                     uint threadgroup_size [[threads_per_threadgroup]])
{
    topLeftX[tid] = INFINITY;
    topLeftY[tid] = -INFINITY;
    bottomRightX[tid] = -INFINITY;
    bottomRightY[tid] = INFINITY;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid < uint(nBodies)) {
        Body body = bodies[gid];
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
        float paddingX = max(abs(rangeX) * 0.01, 0.1);
        float paddingY = max(abs(rangeY) * 0.01, 0.1);
        
        node[0].topLeft.x = fmin(node[0].topLeft.x, topLeftX[0] - paddingX);
        node[0].topLeft.y = fmax(node[0].topLeft.y, topLeftY[0] + paddingY);
        node[0].bottomRight.x = fmax(node[0].bottomRight.x, bottomRightX[0] + paddingX);
        node[0].bottomRight.y = fmin(node[0].bottomRight.y, bottomRightY[0] - paddingY);
        
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

inline void warpReduce(threadgroup volatile float *totalMass, 
                       threadgroup volatile float2 *centerOfMass, 
                       uint tid)
{
    totalMass[tid] += totalMass[tid + 32];
    centerOfMass[tid].x += centerOfMass[tid + 32].x;
    centerOfMass[tid].y += centerOfMass[tid + 32].y;
    totalMass[tid] += totalMass[tid + 16];
    centerOfMass[tid].x += centerOfMass[tid + 16].x;
    centerOfMass[tid].y += centerOfMass[tid + 16].y;
    totalMass[tid] += totalMass[tid + 8];
    centerOfMass[tid].x += centerOfMass[tid + 8].x;
    centerOfMass[tid].y += centerOfMass[tid + 8].y;
    totalMass[tid] += totalMass[tid + 4];
    centerOfMass[tid].x += centerOfMass[tid + 4].x;
    centerOfMass[tid].y += centerOfMass[tid + 4].y;
    totalMass[tid] += totalMass[tid + 2];
    centerOfMass[tid].x += centerOfMass[tid + 2].x;
    centerOfMass[tid].y += centerOfMass[tid + 2].y;
    totalMass[tid] += totalMass[tid + 1];
    centerOfMass[tid].x += centerOfMass[tid + 1].x;
    centerOfMass[tid].y += centerOfMass[tid + 1].y;
}

inline void computeCenterOfMass(thread Node &curNode,
                                device Body *bodies,
                                threadgroup float *totalMass,
                                threadgroup float2 *centerOfMass,
                                int start,
                                int end,
                                uint tid,
                                uint threadgroup_size)
{
    int total = end - start + 1;
    int sz = (total + threadgroup_size - 1) / threadgroup_size;
    int s = tid * sz + start;
    float M = 0.0;
    float2 R = float2(0.0, 0.0);

    for (int i = s; i < s + sz; ++i) {
        if (i <= end) {
            Body body = bodies[i];
            M += body.mass;
            R.x += body.mass * body.position.x;
            R.y += body.mass * body.position.y;
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
        curNode.centerOfMass = float2(centerOfMass[0].x, centerOfMass[0].y);
    }
}

inline void countBodies(device Body *bodies, 
                        float2 topLeft, 
                        float2 bottomRight, 
                        threadgroup int *count, 
                        int start, 
                        int end, 
                        int nBodies,
                        uint tid,
                        uint threadgroup_size)
{
    if (tid < 4)
        count[tid] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = start + tid; i <= end; i += threadgroup_size) {
        Body body = bodies[i];
        int q = getQuadrant(topLeft, bottomRight, body.position.x, body.position.y);
        atomic_fetch_add_explicit((threadgroup atomic_int*)&count[q - 1], 1, memory_order_relaxed);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void computeOffset(threadgroup int *count,
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

inline void groupBodies(device Body *bodies, 
                        device Body *buffer, 
                        float2 topLeft, 
                        float2 bottomRight, 
                        threadgroup int *count, 
                        int start, 
                        int end, 
                        int nBodies,
                        uint tid,
                        uint threadgroup_size)
{
    threadgroup int *count2 = &count[4];
    for (int i = start + tid; i <= end; i += threadgroup_size)
    {
        Body body = bodies[i];
        int q = getQuadrant(topLeft, bottomRight, body.position.x, body.position.y);
        int dest = atomic_fetch_add_explicit((threadgroup atomic_int*)&count2[q - 1], 1, memory_order_relaxed);
        buffer[dest] = body;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void constructQuadTreeKernel(device Node *node [[buffer(0)]],
                                    device Body *bodies [[buffer(1)]],
                                    device Body *buffer [[buffer(2)]],
                                    constant int &nodeOffset [[buffer(3)]],
                                    constant int &nNodes [[buffer(4)]],
                                    constant int &nBodies [[buffer(5)]],
                                    constant int &leafLimit [[buffer(6)]],
                                    threadgroup int *count [[threadgroup(0)]],
                                    threadgroup float *totalMass [[threadgroup(1)]],
                                    threadgroup float2 *centerOfMass [[threadgroup(2)]],
                                    uint tid [[thread_position_in_threadgroup]],
                                    uint bid [[threadgroup_position_in_grid]],
                                    uint threadgroup_size [[threads_per_threadgroup]])
{
    uint nodeIndex = nodeOffset + bid;

    if (nodeIndex >= uint(nNodes))
        return;

    Node curNode = node[nodeIndex];
    int start = curNode.start, end = curNode.end;
    float2 topLeft = curNode.topLeft, bottomRight = curNode.bottomRight;

    if (start == -1 || end == -1)
        return;

    computeCenterOfMass(curNode, bodies, totalMass, centerOfMass, start, end, tid, threadgroup_size);
    
    // Write back the updated node
    if (tid == 0) {
        node[nodeIndex] = curNode;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (nodeIndex >= uint(leafLimit) || start == end) {
        for (int i = start; i <= end; ++i) {
            buffer[i] = bodies[i];
        }
        return;
    }

    countBodies(bodies, topLeft, bottomRight, count, start, end, nBodies, tid, threadgroup_size);
    computeOffset(count, start, tid);
    groupBodies(bodies, buffer, topLeft, bottomRight, count, start, end, nBodies, tid, threadgroup_size);
    
    if (tid == 0) {
        Node topLNode = node[(nodeIndex * 4) + 2];
        Node topRNode = node[(nodeIndex * 4) + 1];
        Node botLNode = node[(nodeIndex * 4) + 3];
        Node botRNode = node[(nodeIndex * 4) + 4];

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
        node[(nodeIndex * 4) + 1] = topRNode;
        node[(nodeIndex * 4) + 2] = topLNode;
        node[(nodeIndex * 4) + 3] = botLNode;
        node[(nodeIndex * 4) + 4] = botRNode;
        
        node[nodeIndex] = curNode;
    }
}

/*
----------------------------------------------------------------------------------------
COMPUTE FORCE
----------------------------------------------------------------------------------------
*/

inline float getDistance(float2 pos1, float2 pos2) {
    return sqrt(pow(pos1.x - pos2.x, 2) + pow(pos1.y - pos2.y, 2));
}

inline bool isCollide(Body b1, float2 cm) {
    return b1.radius * 2 + COLLISION_TH > getDistance(b1.position, cm);
}

void computeForce(device Node *node,
                  device Body *bodies,
                  int nodeIndex,
                  int bodyIndex,
                  int nNodes,
                  int nBodies,
                  int leafLimit,
                  float width)
{
    int stack[32];
    float widthStack[32];
    int stackIdx = 0;
    
    stack[stackIdx] = nodeIndex;
    widthStack[stackIdx] = width;
    stackIdx++;
    
    Body bi = bodies[bodyIndex];
    
    while (stackIdx > 0) {
        stackIdx--;
        int curIndex = stack[stackIdx];
        float curWidth = widthStack[stackIdx];
        
        if (curIndex >= nNodes || curIndex < 0) {
            continue;
        }
        
        Node curNode = node[curIndex];

        if (curNode.isLeaf) {
            if (curNode.centerOfMass.x != -1) {
                float dist = getDistance(bi.position, curNode.centerOfMass);
                if (dist > 0.01) {
                    float2 rij = curNode.centerOfMass - bi.position;
                    float r = sqrt((rij.x * rij.x) + (rij.y * rij.y) + (E * E));
                    float f = (GRAVITY * bi.mass * curNode.totalMass) / (r * r * r + (E * E));
                    float2 force = {rij.x * f, rij.y * f};

                    bodies[bodyIndex].acceleration.x += (force.x / bi.mass);
                    bodies[bodyIndex].acceleration.y += (force.y / bi.mass);
                }
            }
            continue;
        }

        float sd = curWidth / getDistance(bi.position, curNode.centerOfMass);
        if (sd < THETA) {
            float2 rij = curNode.centerOfMass - bi.position;
            float r = sqrt((rij.x * rij.x) + (rij.y * rij.y) + (E * E));
            float f = (GRAVITY * bi.mass * curNode.totalMass) / (r * r * r + (E * E));
            float2 force = {rij.x * f, rij.y * f};

            bodies[bodyIndex].acceleration.x += (force.x / bi.mass);
            bodies[bodyIndex].acceleration.y += (force.y / bi.mass);
            continue;
        }

        float halfWidth = curWidth / 2;
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

kernel void computeForceKernel(device Node *node [[buffer(0)]],
                               device Body *bodies [[buffer(1)]],
                               constant int &nNodes [[buffer(2)]],
                               constant int &nBodies [[buffer(3)]],
                               constant int &leafLimit [[buffer(4)]],
                               uint gid [[thread_position_in_grid]])
{
    float width = node[0].bottomRight.x - node[0].topLeft.x;
    
    if (gid < uint(nBodies)) {
        device Body& bi = bodies[gid];
        if (bi.isDynamic) {
            bi.acceleration = {0.0, 0.0};
            computeForce(node, bodies, 0, gid, nNodes, nBodies, leafLimit, width);
            bi.velocity.x += bi.acceleration.x * DT;
            bi.velocity.y += bi.acceleration.y * DT;
            bi.position.x += bi.velocity.x * DT;
            bi.position.y += bi.velocity.y * DT;
        }
    }
}

