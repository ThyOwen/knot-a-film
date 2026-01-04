#ifdef __METAL_VERSION__

#include <metal_stdlib>
#include "MetalShaders.h"

using namespace metal;

struct Payload {
    float2 origin[32];
    float2 midpoint[32];
    float2 joint[32];
    float2 intersection[32];
    float2 prevMidpoint[32];
    float2 prevJoint[32];
    bool mask[32];
};

struct PointVertexOut {
    float4 position [[position]];
    float size [[point_size]];
};

struct VertexOut {
    float4 position [[position]];
};

struct PrimOut {
    float3 color;
};

struct FragmentIn {
    VertexOut v;
    PrimOut p;
};

using PointMeshType = metal::mesh<PointVertexOut, PrimOut, 256, 256, metal::topology::point>;
using LineMeshType = metal::mesh<VertexOut, PrimOut, 256, 256, metal::topology::line>;

// Edge member data stores pairs of of bodies and their matching sources sorted
// the terminations are sorted by angle going counter clockwise
// the sources are groupped per body with duplicates (both B -> A and A -> B are stored)

[[object]] void objectShader( constant EdgeMemberData<uint>& edgeTerminationsSorted [[buffer(EDGE_TERMINATIONS_SORTED_IDX)]],
                              constant EdgeMemberData<uint>& edgeSources [[buffer(EDGE_SOURCES_IDX)]],
                              constant EdgeMemberData<uint>& edgeTerminations [[buffer(EDGE_TERMINATIONS_IDX)]],
                          
                              constant BodyMemberData<float2>& bodyPositions [[buffer(BODY_POSITION_IDX)]],
                              constant BodyMemberData<uint>& bodyEdgeOffsets [[buffer(BODY_EDGE_OFFSETS_IDX)]],
                             
                              constant ScreenTransform& transform [[buffer(SCREEN_TRANSFORM_IDX)]],
                             
                              object_data Payload& payload [[payload]],
                              mesh_grid_properties meshGridProperties,
                              
                              ushort tid [[thread_index_in_threadgroup]],
                              uint bid [[threadgroup_position_in_grid]],
                              ushort lane_id [[thread_index_in_simdgroup]],
                              ushort threads_per_threadgroup [[threads_per_threadgroup]],
                              ushort threads_per_simdgroup [[threads_per_simdgroup]]
                                                 
) {
    uint gid = bid * threads_per_threadgroup + tid;

    if (gid >= numEdges)
        return;
    
    uint terminationIdx = edgeTerminationsSorted.data[gid];
    uint sourceIdx = edgeSources.data[gid];
    
    uint prevSourceIdx = simd_shuffle_up(sourceIdx, 1);
    
    bool isStart = (lane_id == 0) || (sourceIdx != prevSourceIdx);// || (gid == numEdges);
    
    float2 terminationPoint = bodyPositions.data[terminationIdx];
    float2 sourcePoint = bodyPositions.data[sourceIdx];

    float2 midpoint = (terminationPoint + sourcePoint) / 2;
    float2 delta = terminationPoint - sourcePoint;

    float width = 0.05;
    float2 perpendicular = float2(-delta.y, delta.x) / length(delta);
    float2 joint = (perpendicular * width);
    
    float2 prevJoint = simd_shuffle_up(joint, 1);
    float2 prevMidpoint = simd_shuffle_up(midpoint, 1);
    float2 prevDelta = simd_shuffle_up(delta, 1);
    
    float2 p1 = prevMidpoint + prevJoint;
    float2 d1 = prevDelta;
    float2 p2 = midpoint - joint;
    float2 d2 = delta;
    
    // Solve for intersection
    float2 p1p2 = p2 - p1;
    float intersectionCrossProduct = d1.x * d2.y - d1.y * d2.x;

    float2 intersection;
    if (abs(intersectionCrossProduct) < 1e-6) { // lines are parallel, -> use midpoint between edges
        intersection = (p1 + p2) / 2;
    } else {
        float t = (p1p2.x * d2.y - p1p2.y * d2.x) / intersectionCrossProduct;
        intersection = p1 + t * d1;
    }
    
    //screenspace
    float2 origin_t = (sourcePoint * transform.scale) + transform.offset;
    float2 midpoint_t = (midpoint * transform.scale) + transform.offset;
    float2 joint_t = (p2 * transform.scale) + transform.offset;
    float2 intersection_t = (intersection * transform.scale) + transform.offset;
    float2 prevJoint_t = (p1 * transform.scale) + transform.offset;
    float2 prevMidpoint_t = (prevMidpoint * transform.scale) + transform.offset;
    
    payload.origin[tid] = origin_t;
    payload.midpoint[tid] = midpoint_t;
    payload.joint[tid] = joint_t;
    payload.intersection[tid] = intersection_t;
    payload.prevJoint[tid] = prevJoint_t;
    payload.prevMidpoint[tid] = prevMidpoint_t;
    payload.mask[tid] = !isStart;
    
    meshGridProperties.set_threadgroups_per_grid(uint3(1, 1, 1));
}


[[mesh]] void meshPointShader( PointMeshType output,
                               const object_data Payload& payload [[payload]],
                               ushort tid [[thread_position_in_threadgroup]],
                               uint bid [[threadgroup_position_in_grid]],
                               ushort lane_id [[thread_index_in_simdgroup]],
                               ushort threads_per_threadgroup [[threads_per_threadgroup]],
                               ushort threads_per_simdgroup [[threads_per_simdgroup]]
) {
    uint base = bid * threads_per_threadgroup;
    uint remaining = numEdges > base ? numEdges - base : 0;
    uint count = min(remaining, (uint)threads_per_threadgroup);
    
    output.set_primitive_count(count);

    PointVertexOut v;
    v.position = float4(payload.origin[tid], 0.0, 1.0);
    v.size = 8.0;
    output.set_vertex(tid, v);
    PrimOut p;
    p.color = float3(0.0, 0.5, 1.0);
    output.set_primitive(tid, p);
    output.set_index(tid, tid);
}


[[mesh]] void meshLineShader( LineMeshType output,
                              const object_data Payload& payload [[payload]],
                              ushort tid [[thread_position_in_threadgroup]],
                              uint bid [[threadgroup_position_in_grid]],
                              ushort lane_id [[thread_index_in_simdgroup]],
                              ushort threads_per_threadgroup [[threads_per_threadgroup]],
                              ushort threads_per_simdgroup [[threads_per_simdgroup]]
) {
    uint base = bid * threads_per_threadgroup;
    uint remaining = numEdges > base ? numEdges - base : 0;
    uint count = min(remaining, (uint)threads_per_threadgroup);

    // 6 line primitives per thread
    output.set_primitive_count(count * 6);

    if (!payload.mask[tid])
        return;

    // 7 vertices per thread
    uint vertBase = tid * 7;

    uint v0 = vertBase + 0; // origin
    uint v1 = vertBase + 1; // midpoint
    uint v2 = vertBase + 2; // joint
    uint v3 = vertBase + 3; // intersection
    uint v4 = vertBase + 4; // prevJoint
    uint v5 = vertBase + 5; // prevMidpoint
    uint v6 = vertBase + 6; // origin (dup)

    VertexOut a, b, c, d, e, f, g;

    a.position = float4(payload.origin[tid],       0.0, 1.0);
    b.position = float4(payload.midpoint[tid],     0.0, 1.0);
    c.position = float4(payload.joint[tid],        0.0, 1.0);
    d.position = float4(payload.intersection[tid], 0.0, 1.0);
    e.position = float4(payload.prevJoint[tid],    0.0, 1.0);
    f.position = float4(payload.prevMidpoint[tid], 0.0, 1.0);
    g.position = float4(payload.origin[tid],       0.0, 1.0);

    output.set_vertex(v0, a);
    output.set_vertex(v1, b);
    output.set_vertex(v2, c);
    output.set_vertex(v3, d);
    output.set_vertex(v4, e);
    output.set_vertex(v5, f);
    output.set_vertex(v6, g);

    uint primBase = tid * 6;

    PrimOut p;
    p.color = float3(0.6, 0.8, 0.9);

    for (uint i = 0; i < 6; ++i)
     output.set_primitive(primBase + i, p);

    // 0: origin -> midpoint
    output.set_index((primBase + 0) * 2 + 0, v0);
    output.set_index((primBase + 0) * 2 + 1, v1);

    // 1: midpoint -> joint
    output.set_index((primBase + 1) * 2 + 0, v1);
    output.set_index((primBase + 1) * 2 + 1, v2);

    // 2: joint -> intersection
    output.set_index((primBase + 2) * 2 + 0, v2);
    output.set_index((primBase + 2) * 2 + 1, v3);
    
    // 3: intersection -> prevJoint
    output.set_index((primBase + 3) * 2 + 0, v3);
    output.set_index((primBase + 3) * 2 + 1, v4);
    
    // 3: prevJoint -> prevMidpoint
    output.set_index((primBase + 4) * 2 + 0, v4);
    output.set_index((primBase + 4) * 2 + 1, v5);

    // 4: prevMidpoint -> origin
    output.set_index((primBase + 5) * 2 + 0, v5);
    output.set_index((primBase + 5) * 2 + 1, v6);
}

fragment float4 fragmentBody(FragmentIn in [[stage_in]],
                             float2 pointCoord [[point_coord]]) {
    return float4(in.p.color, 1.0);
}

#endif

