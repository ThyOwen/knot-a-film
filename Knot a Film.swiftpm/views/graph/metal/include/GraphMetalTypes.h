#pragma once

#include <simd/simd.h>
#ifndef __METAL_VERSION__
    #import <Foundation/Foundation.h>
#endif

#define MAX_CONNECTIONS 512
#define NUM_BODIES 4096
#define MAX_NODES 349525

struct PhysicsParams {
    float gravity;
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
    bool isDynamic;
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
    struct Body bodies[NUM_BODIES];
    uint numBodies;
};

struct PerBodyConnectionsData {
    uint perBodyConnections[MAX_CONNECTIONS];
    uint numConnections;
};

struct ConnectionsData {
    struct PerBodyConnectionsData connections[NUM_BODIES];
    uint numBodiesWithConnections;
};
