#include <metal_stdlib>
#include "SharedWithMetal.h"
using namespace metal;

struct BodyPayload {
    float2 position;
    float2 connections[MAX_CONNECTIONS];
    uint numConnections;
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

[[object]] void objectShader(object_data BodyPayload& payload [[payload]],
                             mesh_grid_properties meshGridProperties,
                             constant BodyData& bodyData [[buffer(0)]],
                             constant ScreenTransform& transform [[buffer(1)]],
                             constant ConnectionsData& connectionsData [[buffer(2)]],
                             uint gid [[thread_position_in_grid]])
{
    if (gid >= bodyData.numBodies) {
        return;
    }
    
    thread Body bi = bodyData.bodies[gid];
    
    float2 biPosition = (bi.position * transform.scale) + transform.offset;
    
    if (connectionsData.numConnections != 0) {
        PerBodyConnectionsData data = connectionsData.connections[gid];
        payload.numConnections = data.numPerBodyConnections;
        for (int i = 0; i < (int)data.numPerBodyConnections; i++) {
            uint idx = data.perBodyConnections[i];
            float2 termination = (bodyData.bodies[idx].position * transform.scale) + transform.offset;
            float2 midpoint = termination - biPosition;
            payload.connections[i] = midpoint;
        }
    }

    payload.position = biPosition;
    
    
    meshGridProperties.set_threadgroups_per_grid(uint3(1, 1, 1));
}


[[mesh]] void meshPointShader(PointMeshType output,
                              const object_data BodyPayload& payload [[payload]],
                              ushort tid [[thread_index_in_threadgroup]])
{
    output.set_primitive_count(1);
    PointVertexOut v;
    v.position = float4(payload.position, 0.0, 1.0);
    v.size = 8.0;
    output.set_vertex(tid, v);
    PrimOut p;
    p.color = float3(0.0, 0.5, 1.0);
    output.set_primitive(tid, p);
    output.set_index(tid, tid);
}


[[mesh]] void meshLineShader(LineMeshType output,
                             const object_data BodyPayload& payload [[payload]],
                             ushort tid [[thread_index_in_threadgroup]])
{
    output.set_primitive_count(payload.numConnections);
    
    

    for (uint i = 0; i < payload.numConnections; i++) {
        // start of the line
        VertexOut v0;
        v0.position = float4(payload.position, 0.0, 1.0);
        output.set_vertex(i * 2, v0);

        // end of the line
        VertexOut v1;
        v1.position = float4(payload.position + payload.connections[i], 0.0, 1.0);
        output.set_vertex(i * 2 + 1, v1);

        PrimOut p;
        p.color = float3(1.0, 0.0, 0.0);
        output.set_primitive(i, p);

        output.set_index(i * 2, i * 2);
        output.set_index(i * 2 + 1, i * 2 + 1);
    }
}

fragment float4 fragmentBody(FragmentIn in [[stage_in]],
                             float2 pointCoord [[point_coord]]) {
    return float4(in.p.color, 1.0);
}
