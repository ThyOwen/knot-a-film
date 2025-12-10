//
//  standard.metal
//  Knot a Film
//
//  Created by Owen O'Malley on 12/6/25.
//
#include <metal_stdlib>
#include "include/GraphMetalTypes.h"

//based on https://github.com/Hsin-Hung/N-body-simulation

using namespace metal;

inline float getDistance(float2 a, float2 b) {
    float2 d = a - b;
    return sqrt(d.x*d.x + d.y*d.y);
}

inline float getDistance(half2 a, half2 b) {
    half2 d = a - b;
    return sqrt(d.x*d.x + d.y*d.y);
}

inline bool isCollide(thread const Body &b1,
                      thread const Body &b2,
                      float collisionThreshold) {
    return (b1.radius + b2.radius + collisionThreshold) > getDistance(b1.position, b2.position);
}

kernel void directSumTiledKernel(device Body *bodies      [[buffer(0)]],
                                 constant int* nBodies    [[buffer(1)]],
                                 uint tid                 [[thread_index_in_threadgroup]],
                                 uint group_id            [[threadgroup_position_in_grid]],
                                 uint gid                 [[thread_position_in_grid]],
                                 uint threadsPerThreadgroup   [[threads_per_threadgroup]])
{
    threadgroup Body tileB[BLOCK_SIZE];

    if (gid >= *nBodies) {
        return;
    }

    Body bi = bodies[gid];

    float fx = 0.0f;
    float fy = 0.0f;

    bi.acceleration.x = 0.0f;
    bi.acceleration.y = 0.0f;

    const uint numTiles = (*nBodies + threadsPerThreadgroup - 1) / threadsPerThreadgroup;

    for (uint tile = 0; tile < numTiles; ++tile) {

        uint idx = tile * threadsPerThreadgroup + tid;
        if (idx < *nBodies) {
            tileB[tid] = bodies[idx];
        } else {
            tileB[tid].position.x = 0.0f;
            tileB[tid].position.y = 0.0f;
            tileB[tid].mass = 0.0f;
            tileB[tid].radius = 0.0f;
            tileB[tid].isDynamic = 0;
            tileB[tid].velocity.x = 0.0f;
            tileB[tid].velocity.y = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint maxB = threadsPerThreadgroup;
        for (uint b = 0; b < maxB; ++b) {
            uint j = tile * threadsPerThreadgroup + b;
            if (j >= *nBodies)
                break;

            Body bj = tileB[b];

            if (bi.isDynamic != 0) {//}&& !isCollide(bi, bj, COLLISION_TH)) {
                float rx = bj.position.x - bi.position.x;
                float ry = bj.position.y - bi.position.y;
                float r = sqrt((rx*rx + ry*ry) + (E * E));
                
                float denom = (r * r) + (E * E);

                if (denom > 0.0f) {
                    float f = (GRAVITY * bi.mass * bj.mass) / denom;
                    fx += (rx * f) / bi.mass;
                    fy += (ry * f) / bi.mass;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    bi.acceleration.x += fx;
    bi.acceleration.y += fy;

    bi.velocity.x -= bi.acceleration.x * DT;
    bi.velocity.y -= bi.acceleration.y * DT;

    bi.position.x += bi.velocity.x * DT;
    bi.position.y += bi.velocity.y * DT;

    // write back
    bodies[gid] = bi;
}
