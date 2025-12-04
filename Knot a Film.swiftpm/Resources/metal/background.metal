//
//  background.metal
//  Knot a Film
//
//  Created by Owen O'Malley on 2/1/25.
//

#include <metal_stdlib>
#include <SwiftUI/SwiftUI_Metal.h>
using namespace metal;

/// A simple function that attempts to generate a random number based on various
/// fixed input parameters.
/// - Parameter offset: A fixed value that controls pseudorandomness.
/// - Parameter position: The position of the pixel we're working with.
/// - Parameter time: The number of elapsed seconds since the shader was created.
/// - Returns: The original pixel color.
float whiteRandom(float offset, float2 position) {
    // Pick two numbers that are unlikely to repeat.
    float2 nonRepeating = float2(0.129898, 0.78233);

    // Multiply our texture coordinates by the
    // non-repeating numbers, then add them together.
    float sum = dot(position, nonRepeating);

    // Calculate the sine of our sum to get a range
    // between -1 and 1.
    float sine = sin(sum);

    // Multiply the sine by a big, non-repeating number
    // so that even a small change will result in a big
    // color jump.
    float hugeNumber = sine * 43758.5453 * offset;

    // Send back just the numbers after the decimal point.
    return fract(hugeNumber);
}


half4 sigmoid(half4 pixel, float offest) {
    return divide(1, 1 + exp2(-16 * (pixel - 0.5 + offest)));
}
/// A shader that generates dynamic, grayscale noise.

/// - Parameter position: The user-space coordinate of the current pixel.
/// - Parameter color: The current color of the pixel.
/// - Parameter time: The number of elapsed seconds since the shader was created
/// - Returns: The new pixel color.
[[ stitchable ]] half4 coloredNoise(float2 position,
                                    half4 color,
                                    float strength) {
    // If it's not transparent…
    if (color.a > 0.0h) {
        // Make a color where the RGB values are the same
        // random number and A is 1; multiply by the
        // original alpha to get smooth edges.
        
        float offset = (strength * whiteRandom(1.0, position));
        
        //half3 subColor = half3(color.r, color.g, color.b);
        
        half4 output = sigmoid(color, offset);
        
        return output;
    } else {
        // Use the current (transparent) color.
        return color;
    }
}
