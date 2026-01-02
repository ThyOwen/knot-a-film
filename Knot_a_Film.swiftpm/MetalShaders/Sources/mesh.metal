#ifdef __METAL_VERSION__

#include <metal_stdlib>
#include "MetalShaders.h"

using namespace metal;

struct Payload {
    float2 origin[32];
    float2 midpoint[32];
    float2 extra[32];
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
    
    bool isStart = (gid == 0) || (lane_id == 0) || (sourceIdx != prevSourceIdx);
    
    float2 terminationPoint = bodyPositions.data[terminationIdx];
    float2 sourcePoint = bodyPositions.data[sourceIdx];

    float2 midpoint = (terminationPoint + sourcePoint) / 2;
    
    float2 extra = midpoint + 1.0;
    
    float2 midpoint_t = (midpoint * transform.scale) + transform.offset;
    float2 origin_t = (sourcePoint * transform.scale) + transform.offset;
    float2 extra_t = (extra * transform.scale) + transform.offset;
    
    payload.midpoint[tid] = midpoint_t;
    payload.origin[tid] = origin_t;
    payload.extra[tid] = extra_t;
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
    
    output.set_primitive_count(count * 2);

    if (payload.mask[tid]) {
        uint vertBase = tid * 4;
        
        uint v0 = vertBase + 0; // origin
        uint v1 = vertBase + 1; // midpoint
        uint v2 = vertBase + 2; // midpoint
        uint v3 = vertBase + 3; // extra point
        
        VertexOut a, b, c, d;
        
        a.position = float4(payload.origin[tid],   0.0, 1.0);
        b.position = float4(payload.midpoint[tid], 0.0, 1.0);
        c.position = float4(payload.midpoint[tid], 0.0, 1.0);
        d.position = float4(payload.extra[tid],    0.0, 1.0);
        
        output.set_vertex(v0, a);
        output.set_vertex(v1, b);
        output.set_vertex(v2, c);
        output.set_vertex(v3, d);
        
        uint primBase = tid * 2; // 2 primitives per thread

        PrimOut p;
        p.color = float3(0.6, 0.8, 0.9);
        
        output.set_primitive(primBase + 0, p); // origin → midpoint
        output.set_primitive(primBase + 1, p); // midpoint → extra
        
        // Line 0
        output.set_index((primBase + 0) * 2 + 0, v0);
        output.set_index((primBase + 0) * 2 + 1, v1);
        
        // Line 1
        output.set_index((primBase + 1) * 2 + 0, v2);
        output.set_index((primBase + 1) * 2 + 1, v3);
    }
}

fragment float4 fragmentBody(FragmentIn in [[stage_in]],
                             float2 pointCoord [[point_coord]]) {
    return float4(in.p.color, 1.0);
}

#endif
