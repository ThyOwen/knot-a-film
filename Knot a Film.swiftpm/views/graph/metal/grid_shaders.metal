//
//  vertex.metal
//  Knot a Film
//
//  Created by Owen O'Malley on 2/17/25.
//

#include <metal_stdlib>
#include "SharedWithMetal.h"

using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float  pointSize [[point_size]];
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

fragment float4 fragmentNode(float2 pointCoord [[point_coord]]) {
    return float4(0.8, 0.2, 0.2, 1.0);
}
