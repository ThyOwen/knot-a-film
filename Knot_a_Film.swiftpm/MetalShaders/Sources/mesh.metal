#ifdef __METAL_VERSION__

#include <metal_stdlib>
#include "MetalShaders.h"

using namespace metal;

struct Payload {
    float2 origin[32];
    float2 midpoint[32];
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

[[object]] void objectShader( constant EdgeMemberData<uint>& edgeTerminationsSorted [[buffer(EDGE_TERMINATIONS_SORTED_IDX)]],
                              constant EdgeMemberData<uint>& edgeSources [[buffer(EDGE_SOURCES_IDX)]],
                          
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

    float2 terminationPoint = bodyPositions.data[terminationIdx];
    float2 sourcePoint = bodyPositions.data[sourceIdx];

    float2 midpoint = (terminationPoint + sourcePoint) / 2;
    
    float2 midpoint_t = (midpoint * transform.scale) + transform.offset;
    float2 origin_t = (sourcePoint * transform.scale) + transform.offset;
    
    payload.midpoint[tid] = midpoint_t;
    payload.origin[tid] = origin_t;
    
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
    
    output.set_primitive_count(count);

    uint v0 = tid * 2;
    uint v1 = tid * 2 + 1;
    
    // Start of the line
    VertexOut start;
    start.position = float4(payload.origin[tid], 0.0, 1.0);
    output.set_vertex(v0, start);
    
    // End of the line
    VertexOut midpoint;
    midpoint.position = float4(payload.midpoint[tid], 0.0, 1.0);
    output.set_vertex(v1, midpoint);
    
    PrimOut p;
    p.color = float3(0.6, 0.8, 0.9);
    output.set_primitive(tid, p);
    
    output.set_index(v0, v0);
    output.set_index(v1, v1);
}

fragment float4 fragmentBody(FragmentIn in [[stage_in]],
                             float2 pointCoord [[point_coord]]) {
    return float4(in.p.color, 1.0);
}

#endif
