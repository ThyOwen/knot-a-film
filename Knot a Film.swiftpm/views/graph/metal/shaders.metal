//
//  vertex.metal
//  Knot a Film
//
//  Created by Owen O'Malley on 2/17/25.
//

#include <metal_stdlib>

#include "include/GraphMetalTypes.h"
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float  pointSize [[point_size]];
};

struct ScreenTransform {
    float2 offset;
    float2 scale;
};

vertex VertexOut vertexNode(device const NodeData& nodeData [[ buffer(0)]],
                            constant ScreenTransform& transform [[ buffer(1) ]],
                            uint vertexId [[ vertex_id ]]) {
    //Node node = nodes[vid];
    
    VertexOut out;
    
    uint nodeIndex = vertexId / 8;
    uint corner    = vertexId % 8;
    
    Node node = nodeData.nodes[nodeIndex];
    
    float2 topLeft = node.topLeft;
    float2 bottomRight = node.bottomRight;
    
    float2 topRight = { node.bottomRight.x, node.topLeft.y };
    float2 bottomLeft = { node.topLeft.x, node.bottomRight.y };

    // The 2 triangles that make a box
    float2 corners[8] = {
        topLeft, topRight,
        topRight, bottomRight,
        bottomRight, bottomLeft,
        bottomLeft, topLeft
    };

    float2 pos = (float2(corners[corner]) * transform.scale) + transform.offset;
    //float2 pos = (node.centerOfMass * transform.scale) + transform.offset;
    out.position = float4(pos, 0.0, 1.0);

    return out;
}

vertex VertexOut vertexBody(device const BodyData& bodyData [[ buffer(0)]],
                            constant ScreenTransform& transform [[buffer(1)]],
                            uint vertexId [[ vertex_id ]]) {
    Body body = bodyData.bodies[vertexId];
    VertexOut out;
    
    float2 position = (float2(body.position) * transform.scale) + transform.offset;
    
    out.position = float4(position, 0.0, 1.0);
    out.pointSize = 2.0;   // 12x12 pixels per point
    return out;
}

fragment float4 fragmentNode(float2 pointCoord [[point_coord]]) {
    return float4(0.8, 0.2, 0.2, 1.0);
}

fragment float4 fragmentBody(float2 pointCoord [[point_coord]]) {
    float2 centered = pointCoord * 2.0 - 1.0;
    float dist = length(centered);

    if (dist > 1.0) {
        discard_fragment(); // outside circle
    }

    return float4(1.0, 0.2, 0.2, 1.0);
}


