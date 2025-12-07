#pragma once

#include <simd/simd.h>
#ifndef __METAL_VERSION__
    #import <Foundation/Foundation.h>
#endif

// Constants
#define NUM_BODIES 300
#define WINDOW_WIDTH 2048
#define WINDOW_HEIGHT 2048
#define NBODY_WIDTH 2.0
#define NBODY_HEIGHT 2.0
#define GRAVITY 0.0001
#define E 0.01
#define DT 0.016
#define THETA 0.5
#define CENTERX 0
#define CENTERY 0
#define BLOCK_SIZE 512
#define GRID_SIZE 512
#define MAX_N 4194304
#define MAX_NODES 349525
#define MAX_DEPTH 3
#define N_LEAF 262144
#define COLLISION_TH 0.0001
#define MIN_DIST 0.3
#define MAX_DIST 0.8
#define SUN_MASS 10.0
#define SUN_DIA 0.05
#define EARTH_MASS 1.0
#define EARTH_DIA 0.01
#define HBL 1.6e29


struct Node {
    simd_float2 topLeft;
    simd_float2 bottomRight;
    simd_float2 centerOfMass;
    float totalMass;
    uint start;
    uint end;
    bool isLeaf;
};

struct Body {
    bool isDynamic;
    float mass;
    float radius;
    simd_float2 position;
    simd_float2 velocity;
    simd_float2 acceleration;
};
