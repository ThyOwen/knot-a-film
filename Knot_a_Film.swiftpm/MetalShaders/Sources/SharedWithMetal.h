#pragma once

#include <simd/simd.h>

#ifndef __METAL_VERSION__
    #include <cstdint>
#endif

#define SIMDGROUP_SIZE 32

#define MAX_CONNECTIONS 1024
#define MAX_BODIES 1024
#define MAX_NODES 349525

//MARK: Function Constant Indices

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

//MARK: Node Buffer Indices

#define NODE_TOP_LEFT_IDX           0
#define NODE_BOTTOM_RIGHT_IDX       1
#define NODE_CENTER_OF_MASS_IDX     2
#define NODE_TOTAL_MASS_IDX         3
#define NODE_START_IDX              4
#define NODE_END_IDX                5
#define NODE_IS_LEAF_IDX            6

//MARK: Body Buffer Indices

#define BODY_MASS_IDX               7
#define BODY_RADIUS_IDX             8
#define BODY_POSITION_IDX           9
#define BODY_VELOCITY_IDX           10
#define BODY_ACCELERATION_IDX       11
#define BODY_INITIAL_IDX_IDX        12

#define BODY_MASS_ALT_IDX           13
#define BODY_RADIUS_ALT_IDX         14
#define BODY_POSITION_ALT_IDX       15
#define BODY_VELOCITY_ALT_IDX       16
#define BODY_ACCELERATION_ALT_IDX   17
#define BODY_INITIAL_IDX_ALT_IDX    18

//MARK: Other Buffer Indices

#define CONNECTIONS_IDX             19
#define MUTEX_IDX                   20
#define SCREEN_TRANSFORM_IDX        21
#define PHYSICS_PARAMS_IDX          22

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
    uint32_t maxDepth;
    uint32_t numBodies;
    uint32_t numConnections;
    uint32_t numNodes;
    uint32_t leafLimit;

    int blockSize;
    bool useBarnes;
    bool computeConnections;

    struct PhysicsParams physics;
};

template <typename T>
struct NodeMemberData {
    T data[MAX_NODES];
    uint32_t numInstances;
};

template <typename T>
struct BodyMemberData {
    T data[MAX_BODIES];
    uint32_t numInstances;
};

template struct BodyMemberData<float>;
template struct BodyMemberData<simd_float2>;
template struct BodyMemberData<uint32_t>;

template struct NodeMemberData<simd_float2>;
template struct NodeMemberData<float>;
template struct NodeMemberData<uint32_t>;
template struct NodeMemberData<bool>;

using BodyMemberDataFloat = BodyMemberData<float>;
using BodyMemberDataFloat2 = BodyMemberData<simd_float2>;
using BodyMemberDataUInt32 = BodyMemberData<uint32_t>;

using NodeMemberDataFloat2 = NodeMemberData<simd_float2>;
using NodeMemberDataFloat = NodeMemberData<float>;
using NodeMemberDataUInt32 = NodeMemberData<uint32_t>;
using NodeMemberDataBool = NodeMemberData<bool>;

struct ConnectionsData {
    //uint32_t connections[MAX_BODIES * MAX_CONNECTIONS];
    uint32_t connections[MAX_CONNECTIONS];
    uint32_t offsets[MAX_BODIES + 1];
    uint32_t numConnections;
};
