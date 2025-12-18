#include <metal_stdlib>
#include "include/GraphMetalTypes.h"
using namespace metal;

struct BodyPayload {
    float2 position;
};

struct VertexOut {
    float4 position [[position]];
    float size [[point_size]]; // Optional: set point size
};

struct PrimOut {
    float3 color;
};

struct FragmentIn {
    VertexOut v;
    PrimOut p;
};

using PointMeshType = metal::mesh<VertexOut, PrimOut, 256, 256, metal::topology::point>;

[[object]] void objectShader(object_data BodyPayload& payload [[payload]],
                  mesh_grid_properties meshGridProperties,
                  constant BodyData& bodyData [[buffer(0)]],
                  constant ScreenTransform& transform [[buffer(1)]],
                  uint tid [[thread_position_in_grid]])
{
    if (tid >= bodyData.numBodies) return;
    payload.position = (float2(bodyData.bodies[tid].position) * transform.scale) + transform.offset;
    meshGridProperties.set_threadgroups_per_grid(uint3(1, 1, 1));
}

[[mesh]] void meshShader(PointMeshType output,
                const object_data BodyPayload& payload [[payload]],
                uint tid [[thread_index_in_threadgroup]])
{
    output.set_primitive_count(1);
    VertexOut v;
    v.position = float4(payload.position, 0.0, 1.0);
    v.size = 8.0; 
    output.set_vertex(tid, v);
    PrimOut p;
    p.color = float3(0.0, 0.5, 1.0);
    output.set_primitive(tid, p);
    output.set_index(tid, tid);
}

fragment float4 fragmentBody(FragmentIn in [[stage_in]],
                             float2 pointCoord [[point_coord]]) {
    float2 centered = pointCoord - 0.5;
    float dist = length(centered);

    if (dist > 0.5) {
        discard_fragment();
    }

    return float4(in.p.color, 1.0);
}


