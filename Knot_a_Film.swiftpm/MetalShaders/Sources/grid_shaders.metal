#if defined(__METAL_VERSION__)

#include <metal_stdlib>
#include "SharedWithMetal.h"

using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float  pointSize [[point_size]];
};

vertex VertexOut vertexNode(constant NodeMemberData<float2>& topLeftData [[buffer(NODE_TOP_LEFT_IDX)]],
                            constant NodeMemberData<float2>& bottomRightData [[buffer(NODE_BOTTOM_RIGHT_IDX)]],
                            constant ScreenTransform& transform [[ buffer(SCREEN_TRANSFORM_IDX) ]],
                            uint vertexId [[ vertex_id ]]) {
    //Node node = nodes[vid];
    
    VertexOut out;
    
    uint nodeIndex = vertexId / 8;
    uint corner    = vertexId % 8;
        
    float2 topLeft = topLeftData.data[nodeIndex];
    float2 bottomRight = bottomRightData.data[nodeIndex];
    
    float2 topRight = { bottomRight.x, topLeft.y };
    float2 bottomLeft = { topLeft.x, bottomRight.y };

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

#endif
