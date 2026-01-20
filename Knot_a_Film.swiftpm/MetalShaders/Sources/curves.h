#pragma once

#include <metal_stdlib>

using namespace metal;

float2 cubicBezierCollapsed(float t, float2 p0, float2 p1, float2 p2) {
    float u = 1.0 - t;

    float u2 = u * u;
    float t2 = t * t;

    return
        u2 * u * p0 +
        3.0 * u2 * t * p1 +
        (3.0 * u * t2 + t2 * t) * p2;
}


//MARK: - Conic
float2 conicBezier(float t, float2 p0, float2 p1, float2 p2, float w) {
    float u = 1.0 - t;

    float u2 = u * u;
    float t2 = t * t;

    float b0 = u2;
    float b1 = 2.0 * w * u * t;
    float b2 = t2;

    float denom = b0 + b1 + b2;

    return (b0 * p0 + b1 * p1 + b2 * p2) / denom;
}

float2 conicBezierDerivative(float t, float2 p0, float2 p1, float2 p2, float w) {
    float u = 1.0 - t;
    
    // basis functions
    float b0 = u * u;
    float b1 = 2.0 * w * u * t;
    float b2 = t * t;
    
    // derivatives of basis functions
    float db0 = -2.0 * u;
    float db1 = 2.0 * w * (1.0 - 2.0 * t);
    float db2 = 2.0 * t;
    
    float2 numerator = b0 * p0 + b1 * p1 + b2 * p2;
    float denom = b0 + b1 + b2;
    
    float2 numeratorDerivative = db0 * p0 + db1 * p1 + db2 * p2;
    float denomDerivative = db0 + db1 + db2;
    
    // (n/d)' = (n'd - nd') / d^2
    return (numeratorDerivative * denom - numerator * denomDerivative) / (denom * denom);
}

float2 conicBezierSecondDerivative(float t, float2 p0, float2 p1, float2 p2, float w) {
    float u = 1.0 - t;
    
    // basis functions
    float b0 = u * u;
    float b1 = 2.0 * w * u * t;
    float b2 = t * t;
    
    // first derivatives of basis functions
    float db0 = -2.0 * u;
    float db1 = 2.0 * w * (1.0 - 2.0 * t);
    float db2 = 2.0 * t;
    
    // second derivatives of basis functions
    float ddb0 = 2.0;
    float ddb1 = -4.0 * w;
    float ddb2 = 2.0;
    
    float2 n = b0 * p0 + b1 * p1 + b2 * p2;
    float d = b0 + b1 + b2;
    
    float2 dn = db0 * p0 + db1 * p1 + db2 * p2;
    float dd = db0 + db1 + db2;
    
    float2 ddn = ddb0 * p0 + ddb1 * p1 + ddb2 * p2;
    float ddd = ddb0 + ddb1 + ddb2;
    
    // (n/d)'' = (n''d^2 - 2n'd·d' - nd'^2 + 2nd·d'') / d^3
    float d2 = d * d;
    float d3 = d2 * d;
    
    return (ddn * d2 - 2.0 * dn * d * dd - n * dd * dd + 2.0 * n * d * ddd) / d3;
}
