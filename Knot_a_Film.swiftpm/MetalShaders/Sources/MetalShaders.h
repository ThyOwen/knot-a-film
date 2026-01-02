#pragma once

#include <simd/simd.h>

#ifndef __METAL_VERSION__
    #include <cstdint>
#endif

#define MAX_EDGES 1024
#define MAX_BODIES 1024
#define MAX_NODES 349525

// MARK: Function Constant Indices

#define FC_NUM_BODIES           0
#define FC_NUM_EDGES            1
#define FC_NUM_NODES            2
#define FC_LEAF_LIMIT           3
#define FC_BLOCK_SIZE           4
#define FC_USE_BARNES           6
#define FC_COMPUTE_EDGES        7

#define FC_SPRING_CONSTANT      8
#define FC_EDGE_REPULSION       9
#define FC_EDGE_ATTRACTION      10
#define FC_EPSILON              11
#define FC_DT                   12
#define FC_THETA                13
#define FC_COLLISION_THRESHOLD  14
#define FC_DAMPING              15

// MARK: Node Buffer Indices

#define NODE_TOP_LEFT_IDX           0
#define NODE_BOTTOM_RIGHT_IDX       1
#define NODE_CENTER_OF_MASS_IDX     2
#define NODE_TOTAL_MASS_IDX         3
#define NODE_START_IDX              4
#define NODE_END_IDX                5
#define NODE_IS_LEAF_IDX            6

// MARK: Body Buffer Indices

#define BODY_MASS_IDX               7
#define BODY_RADIUS_IDX             8
#define BODY_POSITION_IDX           9
#define BODY_VELOCITY_IDX           10
#define BODY_ACCELERATION_IDX       11
#define BODY_INITIAL_IDX_IDX        12
#define BODY_EDGE_OFFSETS_IDX       13

#define BODY_MASS_ALT_IDX           14
#define BODY_POSITION_ALT_IDX       15
#define BODY_INITIAL_IDX_ALT_IDX    16

// MARK: Edges Buffer Indicies

#define EDGE_TERMINATIONS_IDX        17
#define EDGE_SOURCES_IDX             18
#define EDGE_ANGLES_IDX              19
#define EDGE_TERMINATIONS_SORTED_IDX 20

// MARK: Other Buffer Indices

#define MUTEX_IDX                   21
#define SCREEN_TRANSFORM_IDX        22
#define PHYSICS_PARAMS_IDX          23

//MARK: Other

#define ELEMENTS_PER_THREAD_INTIALIZE_EDGES 4


// MARK: Generic
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
    uint32_t numEdges;
    uint32_t numNodes;
    uint32_t leafLimit;

    int blockSize;
    bool useBarnes;
    bool computeEdges;

    struct PhysicsParams physics;
};

template <size_t ArraySize, typename T>
struct ArrayData {
    uint32_t size;
    T data[ArraySize];
};

// MARK: Nodes
template <typename T>
using NodeMemberData = ArrayData<MAX_NODES, T>;

template struct ArrayData<MAX_NODES, simd_float2>;
template struct ArrayData<MAX_NODES, float>;
template struct ArrayData<MAX_NODES, uint32_t>;
template struct ArrayData<MAX_NODES, bool>;

using NodeMemberDataFloat2 = NodeMemberData<simd_float2>;
using NodeMemberDataFloat = NodeMemberData<float>;
using NodeMemberDataUInt32 = NodeMemberData<uint32_t>;
using NodeMemberDataBool = NodeMemberData<bool>;

// MARK: Bodies
template <typename T>
using BodyMemberData = ArrayData<MAX_BODIES, T>;

template struct ArrayData<MAX_BODIES, float>;
template struct ArrayData<MAX_BODIES, simd_float2>;
template struct ArrayData<MAX_BODIES, uint32_t>;

using BodyMemberDataFloat = BodyMemberData<float>;
using BodyMemberDataFloat2 = BodyMemberData<simd_float2>;
using BodyMemberDataUInt32 = BodyMemberData<uint32_t>;

// MARK: Edges
template <typename T>
using EdgeMemberData = ArrayData<MAX_BODIES * MAX_EDGES, T>;

template struct ArrayData<MAX_BODIES * MAX_EDGES, uint32_t>;

using EdgeMemberDataUInt32 = EdgeMemberData<uint32_t>;

#ifdef __METAL_VERSION__

constant uint   numBodies          [[ function_constant(FC_NUM_BODIES) ]];
constant uint   numEdges           [[ function_constant(FC_NUM_EDGES) ]];
constant uint   numNodes           [[ function_constant(FC_NUM_NODES) ]];
constant uint   leafLimit          [[ function_constant(FC_LEAF_LIMIT) ]];

constant int   blockSize          [[ function_constant(FC_BLOCK_SIZE) ]];
constant bool  useBarnes          [[ function_constant(FC_USE_BARNES) ]];
constant bool  computeEdges       [[ function_constant(FC_COMPUTE_EDGES) ]];

constant float springConstant     [[ function_constant(FC_SPRING_CONSTANT) ]];
constant float edgeRepulsion      [[ function_constant(FC_EDGE_REPULSION) ]];
constant float edgeAttraction     [[ function_constant(FC_EDGE_ATTRACTION) ]];
constant float epsilon            [[ function_constant(FC_EPSILON) ]];
constant float dt                 [[ function_constant(FC_DT) ]];
constant float theta              [[ function_constant(FC_THETA) ]];
constant float collisionThreshold [[ function_constant(FC_COLLISION_THRESHOLD) ]];
constant float damping            [[ function_constant(FC_DAMPING) ]];

#endif
