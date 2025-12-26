#pragma once

#include <simd/simd.h>
#ifndef __METAL_VERSION__
    #import <Foundation/Foundation.h>
#endif

#define MAX_CONNECTIONS 512
#define MAX_BODIES 1024
#define MAX_NODES 349525

#define FC_NUM_BODIES           0
#define FC_NUM_CONNECTIONS      1
#define FC_NUM_NODES            2
#define FC_LEAF_LIMIT           3
#define FC_BLOCK_SIZE           4
#define FC_USE_BARNES           6
#define FC_COMPUTE_CONNECTIONS  7

#define FC_SPRING_CONSTANT      8
#define FC_EDGE_REPULSION       9
#define FC_EDGE_ATTRACTION      10
#define FC_EPSILON              11
#define FC_DT                   12
#define FC_THETA                13
#define FC_COLLISION_THRESHOLD  14
#define FC_DAMPING              15

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

struct GraphParams {
    uint maxDepth;
    uint numBodies;
    uint numConnections;
    uint numNodes;
    uint leafLimit;

    int blockSize;
    bool useBarnes;
    bool computeConnections;

    struct PhysicsParams physics;
};

//Nodes
struct Node {
    simd_float2 topLeft;
    simd_float2 bottomRight;
    simd_float2 centerOfMass;
    
    float totalMass;
    uint start;
    uint end;
    bool isLeaf;
};

struct NodeData {
    struct Node nodes[MAX_NODES];
    uint numNodes;
};

//Bodies
struct Body {
    float mass;
    float radius;
    simd_float2 position;
    simd_float2 velocity;
    simd_float2 acceleration;
    uint initialIdx;
};

struct BodyData {
    struct Body bodies[MAX_BODIES];
    uint numBodies;
};

//Connections
struct PerBodyConnectionsData {
    uint perBodyConnections[MAX_CONNECTIONS];
    uint numPerBodyConnections;
};

struct ConnectionsData {
    struct PerBodyConnectionsData connections[MAX_BODIES];
    uint numConnections;
};
