#pragma once

#include <simd/simd.h>
#ifndef __METAL_VERSION__
    #import <Foundation/Foundation.h>
#endif

#define MAX_CONNECTIONS 512
#define MAX_BODIES 1024
#define MAX_NODES 349525

struct ScreenTransform {
    simd_float2 offset;
    simd_float2 scale;
};

struct PhysicsParams {
    float springConstant;
    float edgeRepulsion;
    float edgeAttraction;
    float epsilon;
    float dt;
    float theta;
    float collisionThreshold;
    float damping;
};

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
    float mass;
    float radius;
    simd_float2 position;
    simd_float2 velocity;
    simd_float2 acceleration;
};


struct NodeData {
    struct Node nodes[MAX_NODES];
    uint numNodes;
};

struct BodyData {
    struct Body bodies[MAX_BODIES];
    uint numBodies;
};

struct PerBodyConnectionsData {
    uint perBodyConnections[MAX_CONNECTIONS];
    uint numPerBodyConnections;
};

struct ConnectionsData {
    struct PerBodyConnectionsData connections[MAX_BODIES];
    uint numConnections;
};
