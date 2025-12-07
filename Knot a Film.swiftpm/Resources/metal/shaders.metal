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


//functions
vertex VertexOut vertexMain(device const Body *bodies [[ buffer(0)]],
                            constant ScreenTransform& transform [[buffer(1)]],
                            uint vid [[ vertex_id ]]) {
    Body body = bodies[vid];
    VertexOut out;
    
    float2 position = (body.position * transform.scale) + transform.offset;
    
    out.position = float4(position, 0.0, 1.0);
    out.pointSize = 13.0;   // 12x12 pixels per point
    return out;
}

fragment float4 fragmentMain(float2 pointCoord [[point_coord]]) {
    // Remap (0..1) -> (-1..1)

    float2 centered = pointCoord * 2.0 - 1.0;
    float dist = length(centered);

    if (dist > 1.0) {
        discard_fragment(); // outside circle
    }

    return float4(1.0, 0.2, 0.2, 1.0); // red circle
}

