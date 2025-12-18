//
//  sdf.metal
//  MetalTest
//
//  Created by Owen O'Malley on 12/16/25.
//

#include <metal_stdlib>

using namespace metal;

float unionSDF(float d1, float d2) {
  return min(d1, d2); // Closest surface wins
}

float intersectionSDF(float d1, float d2) {
  return max(d1, d2); // Farthest surface wins
}

float differenceSDF(float d1, float d2) {
  return max(d1, -d2); // Remove d2 from d1
}

float smoothUnion(float d1, float d2, float smoothness) {
  float h = max(smoothness - abs(d1 - d2), 0.0) / smoothness;
  return min(d1, d2) - h * h * smoothness * 0.25;
}

[[ stitchable ]] half4 sdfBooleanOps(
    float2 position,
    half4 color,
    float2 size,
    float smoothness // 0=hard, 0.1=smooth
) {
  float2 uv = position / size;
  float2 centered = uv - 0.5;
  // Define two overlapping shapes
  float circle1 = length(centered - float2(-0.1, 0.0)) - 0.15; // Left circle
  float circle2 = length(centered - float2(0.1, 0.0)) - 0.15; // Right circle
  float result = smoothUnion(circle1, circle2, smoothness); // unionSDF is explained below

  // For better visualization, let's mask the 2 shapes, so any values outside the circle are +1 (white) and all values insdide are black (-1). The easiest way to do this is by using step (or smoothStep).
  result = step(0.1, result);

  return half4(result, result, result, 1.0);
}
