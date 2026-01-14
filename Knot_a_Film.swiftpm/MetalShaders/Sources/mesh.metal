#ifdef __METAL_VERSION__

#include <metal_stdlib>
#include "MetalShaders.h"

using namespace metal;

struct Payload {
    float2 origin[32];
    float2 midpoint[32];
    float2 joint[32];
    float2 intersection[32];
    float2 prevMidpoint[32];
    float2 prevJoint[32];
    uint gid[32];
    uint numEdges;
    uint lod;
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
// threads_per_threadgroup is the simdgroup_size
// LineMeshType needs to accommodate: threads_per_threadgroup * (numCurveLines + 6) vertices/primitives
// the limits of the are 256 for vertices and 512 for prims
// with 32 threads and numCurveLines=2: 32 * (6 + 2) = 256 vertices, 32 * 7 = 224 primitives this is the minimum
// for every lod increase after the minimum only the numCurveLines increases and the "base" (origin, midpoint...) so the next level becomes
// with 32 threads and numCurveLines=8: 32 * 8 = 256 vertices, 32 * 7 = 224 primitives for the next threadgroup

using LineMeshType = metal::mesh<VertexOut, PrimOut, 224, 256, metal::topology::line>;


float3 colorConvert(float h, float s, float v) {
    float c = v * s;
    float x = c * (1.0 - abs(fmod(h * 6.0, 2.0) - 1.0));
    float m = v - c;
    
    float3 rgb;
    if (h < 1.0/6.0) {
        rgb = float3(c, x, 0.0);
    } else if (h < 2.0/6.0) {
        rgb = float3(x, c, 0.0);
    } else if (h < 3.0/6.0) {
        rgb = float3(0.0, c, x);
    } else if (h < 4.0/6.0) {
        rgb = float3(0.0, x, c);
    } else if (h < 5.0/6.0) {
        rgb = float3(x, 0.0, c);
    } else {
        rgb = float3(c, 0.0, x);
    }
    
    return rgb + float3(m, m, m);
}

float2 cubicBezierCollapsed(float t, float2 p0, float2 p1, float2 p2) {
    float u = 1.0 - t;

    float u2 = u * u;
    float t2 = t * t;

    return
        u2 * u * p0 +
        3.0 * u2 * t * p1 +
        (3.0 * u * t2 + t2 * t) * p2;
}

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

// Edge member data stores pairs of of bodies and their matching sources sorted
// the terminations are sorted by angle going counter clockwise
// the sources are groupped per body with duplicates (both B -> A and A -> B are stored)
// The payload contains data for up to 32 edges (the simdgroup size)

[[object]] void objectShader( constant EdgeMemberData<uint>& edgeTerminationsSorted [[buffer(EDGE_TERMINATIONS_SORTED_IDX)]],
                              constant EdgeMemberData<uint>& edgeSources [[buffer(EDGE_SOURCES_IDX)]],
                              constant EdgeMemberData<uint>& edgeTerminations [[buffer(EDGE_TERMINATIONS_IDX)]],
                          
                              constant BodyMemberData<float2>& bodyPositions [[buffer(BODY_POSITION_IDX)]],
                              constant BodyMemberData<uint>& bodyEdgeOffsets [[buffer(BODY_EDGE_OFFSETS_IDX)]],
                             
                              constant ScreenTransform& transform [[buffer(SCREEN_TRANSFORM_IDX)]],
                             
                              object_data Payload& payload [[payload]],
                              mesh_grid_properties meshGridProperties,
                              
                              ushort tid [[thread_index_in_threadgroup]],
                              uint bid [[threadgroup_position_in_grid]],
                              ushort lane_id [[thread_index_in_simdgroup]],
                              ushort threads_per_threadgroup [[threads_per_threadgroup]],
                              ushort threads_per_simdgroup [[threads_per_simdgroup]]
                                                 
) {
    uint gid = bid * threads_per_threadgroup + tid;
    
    // this may solve a buggerusky where the number frames don't resolve to
    // the screen because each thread produces a threadgroup
    // there should be a one to one mapping between the number of threadgroups in the mesh and object shader
    uint base = bid * threads_per_threadgroup;
    uint remaining = (base < numEdges) ? (numEdges - base) : 0;
    uint batchCount = min(remaining, (uint)threads_per_threadgroup);
    
    //get rid of this the addresses pulled by the shuffles are bogulus and have no default value
    //if (gid >= numEdges)
        //return;

    uint sourceIdx = edgeSources.data[gid];
    uint terminationIdx = edgeTerminationsSorted.data[gid];
    
    float2 sourcePoint = bodyPositions.data[sourceIdx];
    float2 terminationPoint = bodyPositions.data[terminationIdx];

    float2 delta = terminationPoint - sourcePoint;
    float2 midpoint = (terminationPoint + sourcePoint) * 0.5f;
    
    float width = 0.025;
    float len = length(delta) + 1e-6; // avoid div by zero
    float2 perpendicular = float2(-delta.y, delta.x) / len;
    float2 joint = perpendicular * width;

    float2 prevMidpoint = simd_shuffle_up(midpoint, 1);
    float2 prevJoint    = simd_shuffle_up(joint, 1);
    uint   prevSource   = simd_shuffle_up(sourceIdx, 1);
    float2 prevDelta    = simd_shuffle_up(delta, 1);

    // seams are breaks in the connectivity of the graph.
    // this could be from the first lane not having a neighbor or changes in the the edge source
    bool isSimdSeam = (lane_id == 0);
    bool isSeam = isSimdSeam || (sourceIdx != prevSource);

    if (isSeam) {
        // to close the loop, the "previous" edge is the last edge of this body.
        
        uint startIdx = bodyEdgeOffsets.data[sourceIdx];
        uint endIdx   = bodyEdgeOffsets.data[sourceIdx + 1];
        
        bool isBodyStart = (gid == startIdx);
        
        uint prevGid;
        if (isBodyStart) {
            prevGid = endIdx - 1; // Wrap to end
        } else {
            prevGid = gid - 1;    // Just previous edge (SIMD boundary case)
        }
        
        uint prevTermIdx = edgeTerminationsSorted.data[prevGid];
        float2 prevTermPos = bodyPositions.data[prevTermIdx];
        
        // Recompute 'prev' geometry manually
        float2 pDelta = prevTermPos - sourcePoint; // source is same
        float2 pMid = (prevTermPos + sourcePoint) * 0.5f;
        float pLen = length(pDelta) + 1e-6;
        float2 pPerp = float2(-pDelta.y, pDelta.x) / pLen;
        
        prevDelta    = pDelta;
        prevMidpoint = pMid;
        prevJoint    = pPerp * width;
    }
    
    float2 p1 = prevMidpoint + prevJoint;
    float2 p2 = midpoint - joint;
    
    float2 d1 = prevDelta;
    float2 d2 = delta;
    
    float2 p1p2 = p2 - p1;
    float crossProduct = d1.x * d2.y - d1.y * d2.x;

    float2 intersection;
    if (abs(crossProduct) < 1e-5) {
        intersection = (p1 + p2) * 0.5f;
    } else {
        float t = (p1p2.x * d2.y - p1p2.y * d2.x) / crossProduct;
        intersection = p1 + t * d1;
    }
    //screenspace
    float2 origin_t = (sourcePoint * transform.scale) + transform.offset;
    float2 midpoint_t = (midpoint * transform.scale) + transform.offset;
    float2 joint_t = (p2 * transform.scale) + transform.offset;
    float2 intersection_t = (intersection * transform.scale) + transform.offset;
    float2 prevJoint_t = (p1 * transform.scale) + transform.offset;
    float2 prevMidpoint_t = (prevMidpoint * transform.scale) + transform.offset;
    
    payload.origin[tid] = origin_t;
    payload.midpoint[tid] = midpoint_t;
    payload.joint[tid] = joint_t;
    payload.intersection[tid] = intersection_t;
    payload.prevJoint[tid] = prevJoint_t;
    payload.prevMidpoint[tid] = prevMidpoint_t;
    payload.gid[tid] = gid;
    
    // Dispatch logic
    // Number of intermediate sample points on the bezier curve (results in numCurveLines + 1 line segments for the curve)
    // Max value depends on mesh limits: must satisfy threads_per_threadgroup * (numCurveLines + 2) <= 256
    // With 32 threads: max numCurveLines = floor(256/32) - 6 = 8
    constexpr int lod = 2; // 1 is the base case
     
    if (tid == 0) {
        payload.numEdges = batchCount;
        payload.lod = lod;
        meshGridProperties.set_threadgroups_per_grid(uint3(lod, 1, 1));
    }
    
}


[[mesh]] void meshPointShader( PointMeshType output,
                               const object_data Payload& payload [[payload]],
                               ushort tid [[thread_position_in_threadgroup]],
                               uint bid [[threadgroup_position_in_grid]],
                               ushort lane_id [[thread_index_in_simdgroup]],
                               ushort threads_per_threadgroup [[threads_per_threadgroup]],
                               ushort threads_per_simdgroup [[threads_per_simdgroup]]
) {
    // Each object threadgroup spawns exactly 1 mesh threadgroup so bid is always 0
    uint count = min((uint)threads_per_threadgroup, numEdges);
    
    if (tid == 0) {
        output.set_primitive_count(count);
    }
    
    //if (tid < count) {
    PointVertexOut v;
    v.position = float4(payload.origin[tid], 0.0, 1.0);
    v.size = 8.0;
    output.set_vertex(tid, v);
    
    // Map gid to hue (0 to 1)
    float hue = (float)payload.gid[tid] / (float)numEdges;
    
    PrimOut p;
    p.color = colorConvert(hue, 0.8, 1.0);
    output.set_primitive(tid, p);
    output.set_index(tid, tid);
    //}
}


[[mesh]] void meshLineShader( LineMeshType output,
                              const object_data Payload& payload [[payload]],
                              ushort tid [[thread_position_in_threadgroup]],
                              uint bid [[threadgroup_position_in_grid]],
                              ushort lane_id [[thread_index_in_simdgroup]],
                              ushort threads_per_threadgroup [[threads_per_threadgroup]],
                              ushort threads_per_simdgroup [[threads_per_simdgroup]]
) {
    // Each object threadgroup spawns exactly 1 mesh threadgroup (bid is always 0)
    // The payload contains data for up to 32 edges (threads_per_threadgroup)
    uint count = min((uint)threads_per_threadgroup, numEdges);

    // 6 line primitives per thread
    if (tid == 0) {
        output.set_primitive_count(count * 6);
    }

    //if (tid >= count || payload.shouldMask[tid])
        //return;

    // 7 vertices per thread
    uint vertBase = tid * 7;

    uint v0 = vertBase + 0; // origin
    uint v1 = vertBase + 1; // midpoint
    uint v2 = vertBase + 2; // joint
    uint v3 = vertBase + 3; // intersection
    uint v4 = vertBase + 4; // prevJoint
    uint v5 = vertBase + 5; // prevMidpoint
    uint v6 = vertBase + 6; // origin (dup)

    VertexOut a, b, c, d, e, f, g;

    a.position = float4(payload.origin[tid],       0.0, 1.0);
    b.position = float4(payload.midpoint[tid],     0.0, 1.0);
    c.position = float4(payload.joint[tid],        0.0, 1.0);
    d.position = float4(payload.intersection[tid], 0.0, 1.0);
    e.position = float4(payload.prevJoint[tid],    0.0, 1.0);
    f.position = float4(payload.prevMidpoint[tid], 0.0, 1.0);
    g.position = float4(payload.origin[tid],       0.0, 1.0);

    output.set_vertex(v0, a);
    output.set_vertex(v1, b);
    output.set_vertex(v2, c);
    output.set_vertex(v3, d);
    output.set_vertex(v4, e);
    output.set_vertex(v5, f);
    output.set_vertex(v6, g);

    uint primBase = tid * 6;

    float hue = float(payload.gid[tid]) / float(numEdges);
    
    PrimOut p;
    p.color = colorConvert(hue, 0.8, 1.0);

    for (uint i = 0; i < 6; ++i)
     output.set_primitive(primBase + i, p);

    // 0: origin -> midpoint
    output.set_index((primBase + 0) * 2 + 0, v0);
    output.set_index((primBase + 0) * 2 + 1, v1);

    // 1: midpoint -> joint
    output.set_index((primBase + 1) * 2 + 0, v1);
    output.set_index((primBase + 1) * 2 + 1, v2);

    // 2: joint -> intersection
    output.set_index((primBase + 2) * 2 + 0, v2);
    output.set_index((primBase + 2) * 2 + 1, v3);
    
    // 3: intersection -> prevJoint
    output.set_index((primBase + 3) * 2 + 0, v3);
    output.set_index((primBase + 3) * 2 + 1, v4);
    
    // 3: prevJoint -> prevMidpoint
    output.set_index((primBase + 4) * 2 + 0, v4);
    output.set_index((primBase + 4) * 2 + 1, v5);

    // 4: prevMidpoint -> origin
    output.set_index((primBase + 5) * 2 + 0, v5);
    output.set_index((primBase + 5) * 2 + 1, v6);
}

[[mesh]] void meshCurveLineShader( LineMeshType output,
                                   const object_data Payload& payload [[payload]],
                                   ushort tid [[thread_index_in_threadgroup]],
                                   uint bid [[threadgroup_position_in_grid]],
                                   ushort lane_id [[thread_index_in_simdgroup]],
                                   ushort threads_per_threadgroup [[threads_per_threadgroup]],
                                   ushort threads_per_simdgroup [[threads_per_simdgroup]]
) {
    // The payload contains data for up to 32 edges (threads_per_threadgroup)
    // payload.numEdges is the actual count for this object threadgroup's batch
    
    uint count = payload.numEdges;
    
    constexpr int numPointsPerThread = 256 / SIMDGROUP_SIZE;
    
    // Each object threadgroup spawns 'lod' mesh threadgroups (bid 0 to lod-1)
    // bid is local to each object's mesh grid, so bid == 0 is base case
    bool isBaseCase = bid == 0;
    
    // Entry/exit lines: origin->midpoint, midpoint->joint, joint->bezier[0], bezier[last - 1]->bezier[last], bezier[last]->prevJoint, prevJoint->prevMidpoint, prevMidpoint->origin
    // Note: joint->bezier[0] and bezier[last]->prevJoint count as part of the bezier section in indexing

    int numLines = numPointsPerThread - 1; // numCurveLines + (numBaseLines - 1);
    
    if (tid == 0) {
        output.set_primitive_count(count * numLines);
    }
    
    uint primBase = tid * numLines;
    
    float hue = float(payload.gid[tid]) / float(numEdges);
    
    PrimOut p;
    //p.color = colorConvert(hue, 0.8, 1.0);
    p.color = isBaseCase ? float3(0.5, 0.5, 1.0) : colorConvert(hue, 0.8, 1.0);

    if (isBaseCase) {
        // Base case renders: origin->midpoint->joint->firstCurvePoint, bridgePoint->lastCurvePoint->prevJoint->prevMidpoint->origin
        // there is an extra bridge index before the lastCurve point
        uint bezierIndices[3];
        
        uint vertBase = tid * (numLines + 1); // add one vertex to the number of lines for the termination per thread
        
        uint originIdx = vertBase + 0;
        uint midpointIdx = vertBase + 1;
        uint jointIdx = vertBase + 2;
        uint firstCurveIdx = vertBase + 3;
        uint bridgeCurveIdx = vertBase + 4;  // Bridge from last non-base threadgroup
        uint lastCurveIdx = vertBase + 5;
        uint prevJointIdx = vertBase + 6;
        uint prevMidpointIdx = vertBase + 7;
        
        bezierIndices[0] = firstCurveIdx;
        bezierIndices[1] = bridgeCurveIdx;
        bezierIndices[2] = lastCurveIdx;
        
        //VERTEX
        VertexOut originVertex, midpointVertex, jointVertex, prevJointVertex, prevMidpointVertex, originTerminationVertex;
        VertexOut bezierVertices[3];
        
        float2 origin = payload.origin[tid];
        float2 joint = payload.joint[tid];
        float2 intersection = payload.intersection[tid];
        float2 prevJoint = payload.prevJoint[tid];
        
        originVertex.position = float4(origin, 0.0, 1.0);
        midpointVertex.position = float4(payload.midpoint[tid], 0.0, 1.0);
        jointVertex.position = float4(joint, 0.0, 1.0);
        
        prevJointVertex.position = float4(prevJoint, 0.0, 1.0);
        prevMidpointVertex.position = float4(payload.prevMidpoint[tid], 0.0, 1.0);
        originTerminationVertex.position = float4(origin, 0.0, 1.0);
        
        output.set_vertex(originIdx, originVertex);
        output.set_vertex(midpointIdx, midpointVertex);
        output.set_vertex(jointIdx, jointVertex);
        
        output.set_vertex(prevJointIdx, prevJointVertex);
        output.set_vertex(prevMidpointIdx, prevMidpointVertex);
        // Use prevJoint slot for origin termination (reusing vertex slot)
        
        // Calculate total curve divisions across all LOD levels
        // Each non-base-case threadgroup renders (numPointsPerThread - 1) new points (1 vertex is bridge)
        // Base case renders first (point 0) and last (point N-1)
        int newPointsPerNonBase = numPointsPerThread - 1; // 7
        int totalCurvePoints = 2 + ((int)payload.lod - 1) * newPointsPerNonBase; // 2 + (lod-1)*7
        
        // First curve point (index 0)
        float t0 = 1.0f / (float)(totalCurvePoints + 1);
        float2 firstCurvePoint = conicBezier(t0, joint, intersection, prevJoint, 3.0f);
        bezierVertices[0].position = float4(firstCurvePoint, 0.0, 1.0);
        output.set_vertex(firstCurveIdx, bezierVertices[0]);
        
        // Bridge point: last point rendered by the last non-base threadgroup
        // Last non-base (bid = lod-1) renders points up to: 1 + (lod-2)*7 + 6 = (lod-1)*7
        // That equals totalCurvePoints - 2 (since totalCurvePoints = 2 + (lod-1)*7)
        int lastNonBaseLastIdx = totalCurvePoints - 2;
        float tBridge = (float)(lastNonBaseLastIdx + 1) / (float)(totalCurvePoints + 1);
        float2 bridgeCurvePoint = conicBezier(tBridge, joint, intersection, prevJoint, 3.0f);
        bezierVertices[1].position = float4(bridgeCurvePoint, 0.0, 1.0);
        output.set_vertex(bridgeCurveIdx, bezierVertices[1]);
        
        // Last curve point (index totalCurvePoints - 1)
        float tLast = (float)totalCurvePoints / (float)(totalCurvePoints + 1);
        float2 lastCurvePoint = conicBezier(tLast, joint, intersection, prevJoint, 3.0f);
        bezierVertices[2].position = float4(lastCurvePoint, 0.0, 1.0);
        output.set_vertex(lastCurveIdx, bezierVertices[2]);
        
        for (int i = 0; i < numLines; ++i) {
            output.set_primitive(primBase + i, p);
        }
        
        // 0: origin -> midpoint
        output.set_index((primBase + 0) * 2 + 0, originIdx);
        output.set_index((primBase + 0) * 2 + 1, midpointIdx);
        
        // 1: midpoint -> joint
        output.set_index((primBase + 1) * 2 + 0, midpointIdx);
        output.set_index((primBase + 1) * 2 + 1, jointIdx);
        
        // 2: joint -> first_curve_point
        output.set_index((primBase + 2) * 2 + 0, jointIdx);
        output.set_index((primBase + 2) * 2 + 1, firstCurveIdx);
        
        // Gap here filled by non-base threadgroups
        
        // 3: bridge_point -> last_curve_point (connects last non-base output to our last point)
        output.set_index((primBase + 3) * 2 + 0, bridgeCurveIdx);
        output.set_index((primBase + 3) * 2 + 1, lastCurveIdx);
        
        // 4: last_curve_point -> previous_joint
        output.set_index((primBase + 4) * 2 + 0, lastCurveIdx);
        output.set_index((primBase + 4) * 2 + 1, prevJointIdx);
        
        // 5: previous_joint -> previous_midpoint
        output.set_index((primBase + 5) * 2 + 0, prevJointIdx);
        output.set_index((primBase + 5) * 2 + 1, prevMidpointIdx);
        
        // 6: previous_midpoint -> origin
        output.set_index((primBase + 6) * 2 + 0, prevMidpointIdx);
        output.set_index((primBase + 6) * 2 + 1, originIdx);
        

    } else {
        // Non-base case: render bridge vertex + (numPointsPerThread - 1) new curve points
        // Total: 8 vertices, 7 lines
        uint bezierIndices[numPointsPerThread]; // 8 vertices
        VertexOut bezierVertices[numPointsPerThread];
        
        uint vertBase = tid * (numLines + 1);
        
        float2 joint = payload.joint[tid];
        float2 intersection = payload.intersection[tid];
        float2 prevJoint = payload.prevJoint[tid];
        
        // Each non-base renders 7 new points (plus 1 bridge = 8 vertices total)
        int newPointsPerNonBase = numPointsPerThread - 1; // 7
        int totalCurvePoints = 2 + ((int)payload.lod - 1) * newPointsPerNonBase;
        
        // curveOffset is the first NEW point index for this threadgroup
        // bid=1: curveOffset = 1 (renders new points 1-7)
        // bid=2: curveOffset = 8 (renders new points 8-14)
        int curveOffset = 1 + ((int)bid - 1) * newPointsPerNonBase;
        
        // Bridge vertex: previous segment's last point (curveOffset - 1)
        int bridgeIdx = curveOffset - 1;
        float bridgeT = (float)(bridgeIdx + 1) / (float)(totalCurvePoints + 1);
        float2 bridgePoint = conicBezier(bridgeT, joint, intersection, prevJoint, 3.0f);
        bezierIndices[0] = vertBase;
        bezierVertices[0].position = float4(bridgePoint, 0.0, 1.0);
        output.set_vertex(bezierIndices[0], bezierVertices[0]);
        
        // Render new curve points: curveOffset to curveOffset + newPointsPerNonBase - 1
        for (int i = 0; i < newPointsPerNonBase; i++) {
            int pointIdx = curveOffset + i;
            float t = (float)(pointIdx + 1) / (float)(totalCurvePoints + 1);
            float2 curvePoint = conicBezier(t, joint, intersection, prevJoint, 3.0f);
            bezierIndices[i + 1] = vertBase + i + 1;
            bezierVertices[i + 1].position = float4(curvePoint, 0.0, 1.0);
            output.set_vertex(bezierIndices[i + 1], bezierVertices[i + 1]);
        }
        
        for (int i = 0; i < numLines; ++i) {
            output.set_primitive(primBase + i, p);
        }
        
        // Draw 7 lines: bridge → p0 → p1 → ... → p6
        for (int i = 0; i < newPointsPerNonBase; i++) {
            output.set_index((primBase + i) * 2 + 0, bezierIndices[i]);
            output.set_index((primBase + i) * 2 + 1, bezierIndices[i + 1]);
        }
    }
}


fragment float4 fragmentBody(FragmentIn in [[stage_in]], float2 pointCoord [[point_coord]]) {
    return float4(in.p.color, 1.0);
}

#endif


