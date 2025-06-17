//
//  vertex.metal
//  Knot a Film
//
//  Created by Owen O'Malley on 2/17/25.
//

#include <metal_stdlib>
using namespace metal;

// Structure to hold the vertex position
struct VertexIn {
    float4 position [[attribute(0)]];
};

struct VertexOut {
    float4 position [[position]];
};

// Vertex shader to transform positions (simple pass-through in this case)
vertex VertexOut vertex_main(VertexIn in [[stage_in]]) {
    VertexOut out;
    out.position = in.position;
    return out;
}

fragment float4 fragment_main() {
    return float4(1.0, 1.0, 1.0, 1.0); // White color
}
