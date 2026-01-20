#ifdef __METAL_VERSION__

#include <metal_stdlib>
#include "MetalShaders.h"
#include "curves.h"

using namespace metal;

struct Payload {
    float2 origin[32];
    
    float2 prevIntersection[32];
    float2 nextIntersection[32];
    
    float2 prevMidpoint[32];
    float2 midpoint[32];
    float2 nextMidpoint[32];
    
    float2 prevJoint[32];
    float2 leftJoint[32];
    float2 rightJoint[32];
    float2 nextJoint[32];
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
    float2 screenPosition;
};

struct PrimOut {
    float3 color;
    
    float2 leftJoint; // leftJoint
    float2 prevIntersection; // prevIntersection
    float2 prevJoint;   // prevJoint
    
    float2 rightJoint;// rightP0
    float2 nextIntersection;// rightP1
    float2 nextJoint;// rightP2
    
    float weight;
    
    // Store t-values for the three vertices of this triangle
    float t0, t1, t2;
};

struct FragmentIn {
    VertexOut v;
    PrimOut p;
};

using PointMeshType = metal::mesh<PointVertexOut, PrimOut, 256, 256, metal::topology::point>;
// threads_per_threadgroup is the simdgroup_size
// LineMeshType needs to accommodate: threads_per_threadgroup * (numCurveLines + 6) vertices/primitives
// each thread will spit out 8 vertices
// vertices, prims
// the limits of the are 256 for vertices and 512 for prims
// with 32 threads and numCurveLines=8: 32 * 8 = 256 vertices, 32 * 6 = 192 primitive


using LineMeshType = metal::mesh<VertexOut, PrimOut, 256, 192, metal::topology::line>;
using TriangleMeshType = metal::mesh<VertexOut, PrimOut, 256, 192, metal::topology::triangle>;



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

// Edge member data stores pairs of of bodies and their matching sources sorted
// the terminations are sorted by angle going counter clockwise
// the sources are groupped per body with duplicates (both B -> A and A -> B are stored)
// The payload contains data for up to 32 edges (the simdgroup size)

[[object]] void objectShaderSimple( constant EdgeMemberData<uint>& edgeTerminationsSorted [[buffer(EDGE_TERMINATIONS_SORTED_IDX)]],
                                    constant EdgeMemberData<uint>& edgeSources [[buffer(EDGE_SOURCES_IDX)]],
                                    constant EdgeMemberData<uint>& edgeTerminations [[buffer(EDGE_TERMINATIONS_IDX)]],
                                    constant BodyMemberData<float2>& bodyPositions [[buffer(BODY_POSITION_IDX)]],
                                    constant BodyMemberData<uint>& bodyEdgeOffsets [[buffer(BODY_EDGE_OFFSETS_IDX)]],
                                    constant ScreenTransform& transform [[buffer(SCREEN_TRANSFORM_IDX)]],
                                    object_data Payload& payload [[payload]],
                                    mesh_grid_properties meshGridProperties,
                                    ushort tid [[thread_index_in_threadgroup]],
                                    uint bid [[threadgroup_position_in_grid]],
                                    ushort threads_per_threadgroup [[threads_per_threadgroup]]
) {
    uint gid = bid * threads_per_threadgroup + tid;
    
    uint base = bid * threads_per_threadgroup;
    uint remaining = (base < numEdges) ? (numEdges - base) : 0;
    uint batchCount = min(remaining, (uint)threads_per_threadgroup);
    
    if (gid >= numEdges) {
        if (tid == 0) {
            payload.numEdges = batchCount;
            payload.lod = 1;
            meshGridProperties.set_threadgroups_per_grid(uint3(0, 1, 1));
        }
        return;
    }

    uint sourceIdx = edgeSources.data[gid];
    uint terminationIdx = edgeTerminationsSorted.data[gid];
    
    float2 sourcePoint = bodyPositions.data[sourceIdx];
    float2 terminationPoint = bodyPositions.data[terminationIdx];

    float2 delta = terminationPoint - sourcePoint;
    float2 midpoint = (terminationPoint + sourcePoint) * 0.5f;
    
    float width = 0.025;
    float len = length(delta) + 1e-6;
    float2 perpendicular = float2(-delta.y, delta.x) / len;
    float2 joint = perpendicular * width;

    uint startIdx = bodyEdgeOffsets.data[sourceIdx];
    uint endIdx = bodyEdgeOffsets.data[sourceIdx + 1];

    uint prevGid = (gid == startIdx) ? (endIdx - 1) : (gid - 1);
    uint nextGid = (gid == endIdx - 1) ? startIdx : (gid + 1);
    
    // previous edge data
    uint prevTermIdx = edgeTerminationsSorted.data[prevGid];
    float2 prevTermPos = bodyPositions.data[prevTermIdx];
    float2 prevDelta = prevTermPos - sourcePoint;
    float2 prevMidpoint = (prevTermPos + sourcePoint) * 0.5f;
    float prevLen = length(prevDelta) + 1e-6;
    float2 prevPerp = float2(-prevDelta.y, prevDelta.x) / prevLen;
    float2 prevJoint = prevPerp * width;
    
    // next edge data
    uint nextTermIdx = edgeTerminationsSorted.data[nextGid];
    float2 nextTermPos = bodyPositions.data[nextTermIdx];
    float2 nextDelta = nextTermPos - sourcePoint;
    float2 nextMidpoint = (nextTermPos + sourcePoint) * 0.5f;
    float nLen = length(nextDelta) + 1e-6;
    float2 nextPerp = float2(-nextDelta.y, nextDelta.x) / nLen;
    float2 nextJoint = nextPerp * width;
    
    // previous intersection
    float2 prevP1 = prevMidpoint + prevJoint;
    float2 prevP2 = midpoint - joint;
    float2 prevP1P2 = prevP2 - prevP1;
    float prevCross = prevDelta.x * delta.y - prevDelta.y * delta.x;
    
    float2 prevIntersection;
    if (abs(prevCross) < 1e-5) {
        prevIntersection = (prevP1 + prevP2) * 0.5f;
    } else {
        float t = (prevP1P2.x * delta.y - prevP1P2.y * delta.x) / prevCross;
        prevIntersection = prevP1 + t * prevDelta;
    }
    
    // next intersection
    float2 nextP1 = midpoint + joint;
    float2 nextP2 = nextMidpoint - nextJoint;
    float2 nextP1P2 = nextP2 - nextP1;
    float nextCross = delta.x * nextDelta.y - delta.y * nextDelta.x;
    
    float2 nextIntersection;
    if (abs(nextCross) < 1e-5) {
        nextIntersection = (nextP1 + nextP2) * 0.5f;
    } else {
        float t = (nextP1P2.x * nextDelta.y - nextP1P2.y * nextDelta.x) / nextCross;
        nextIntersection = nextP1 + t * delta;
    }
    
    // screen space
    float2 origin_t = (sourcePoint * transform.scale) + transform.offset;
    float2 prevIntersection_t = (prevIntersection * transform.scale) + transform.offset;
    float2 nextIntersection_t = (nextIntersection * transform.scale) + transform.offset;
    float2 prevJoint_t = (prevP1 * transform.scale) + transform.offset;
    float2 leftJoint_t = (prevP2 * transform.scale) + transform.offset;
    float2 rightJoint_t = (nextP1 * transform.scale) + transform.offset;
    float2 nextJoint_t = (nextP2 * transform.scale) + transform.offset;
    float2 prevMidpoint_t = (prevMidpoint * transform.scale) + transform.offset;
    float2 midpoint_t = (midpoint * transform.scale) + transform.offset;
    float2 nextMidpoint_t = (nextMidpoint * transform.scale) + transform.offset;

    payload.origin[tid] = origin_t;
    payload.midpoint[tid] = midpoint_t;
    payload.gid[tid] = gid;
    payload.prevIntersection[tid] = prevIntersection_t;
    payload.nextIntersection[tid] = nextIntersection_t;
    payload.prevJoint[tid] = prevJoint_t;
    payload.leftJoint[tid] = leftJoint_t;
    payload.rightJoint[tid] = rightJoint_t;
    payload.nextJoint[tid] = nextJoint_t;
    payload.prevMidpoint[tid] = prevMidpoint_t;
    payload.nextMidpoint[tid] = nextMidpoint_t;

    // Dispatch logic
    constexpr int lod = 2;
     
    if (tid == 0) {
        payload.numEdges = batchCount;
        payload.lod = lod;
        meshGridProperties.set_threadgroups_per_grid(uint3(lod, 1, 1));
    }
}

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
    uint remaining = select(0u, numEdges - base, base < numEdges);
    uint batchCount = min(remaining, (uint)threads_per_threadgroup);
    
    //get rid of this the addresses pulled by the shuffles are bogulus and have no default value
    if (gid >= numEdges) {
        if (tid == 0) {
            //payload.numEdges = batchCount;
            //payload.lod = 1;
            //meshGridProperties.set_threadgroups_per_grid(uint3(0, 1, 1));
        }
        //return;
    }

    uint sourceIdx = edgeSources.data[gid];
    uint terminationIdx = edgeTerminationsSorted.data[gid];
    
    float2 sourcePoint = bodyPositions.data[sourceIdx];
    float2 terminationPoint = bodyPositions.data[terminationIdx];

    float2 delta = terminationPoint - sourcePoint;
    float2 midpoint = (terminationPoint + sourcePoint) * 0.5f;
    
    float width = 0.025;
    float len = length(delta) + 1e-6;
    float2 perpendicular = float2(-delta.y, delta.x) / len;
    float2 joint = perpendicular * width;

    // Get SIMD shuffled values
    float2 simdPrevMidpoint = simd_shuffle_up(midpoint, 1);
    float2 simdNextMidpoint = simd_shuffle_down(midpoint, 1);
    float2 simdPrevJoint = simd_shuffle_up(joint, 1);
    float2 simdNextJoint = simd_shuffle_down(joint, 1);
    uint simdPrevSource = simd_shuffle_up(sourceIdx, 1);
    uint simdNextSource = simd_shuffle_down(sourceIdx, 1);
    float2 simdPrevDelta = simd_shuffle_up(delta, 1);
    float2 simdNextDelta = simd_shuffle_down(delta, 1);
    
    // seams are breaks in the connectivity of the graph.
    // this could be from the first lane not having a neighbor or changes in the the edge source
    bool isSimdUpSeam = (lane_id == 0);
    bool isUpSeam = isSimdUpSeam || (sourceIdx != simdPrevSource);
    
    bool isSimdDownSeam = (lane_id == threads_per_simdgroup - 1);
    bool isDownSeam = isSimdDownSeam || (sourceIdx != simdNextSource);

    uint startIdx = bodyEdgeOffsets.data[sourceIdx];
    uint endIdx = bodyEdgeOffsets.data[sourceIdx + 1];
    
    // Compute wrapped indices to close the loop, the "previous" edge is the last edge of this body.
    bool isBodyStart = (gid == startIdx);
    uint prevGid = select(gid - 1, endIdx - 1, isBodyStart);
    
    // to close the loop, the "next" edge wraps to the first edge of this body.
    bool isBodyEnd = (gid == endIdx - 1);
    uint nextGid = select(gid + 1, startIdx, isBodyEnd);
    
    uint wrappedPrevTermIdx = edgeTerminationsSorted.data[prevGid];
    uint wrappedNextTermIdx = edgeTerminationsSorted.data[nextGid];
    float2 wrappedPrevTermPos = bodyPositions.data[wrappedPrevTermIdx];
    float2 wrappedNextTermPos = bodyPositions.data[wrappedNextTermIdx];
    
    // recompute 'prev' geometry manually
    float2 wrappedPrevDelta = wrappedPrevTermPos - sourcePoint; // source is same
    float2 wrappedPrevMid = (wrappedPrevTermPos + sourcePoint) * 0.5f;
    float wrappedPrevLen = length(wrappedPrevDelta) + 1e-6;
    float2 wrappedPrevPerp = float2(-wrappedPrevDelta.y, wrappedPrevDelta.x) / wrappedPrevLen;
    float2 wrappedPrevJoint = wrappedPrevPerp * width;
    
    // recompute 'next' geometry manually
    float2 wrappedNextDelta = wrappedNextTermPos - sourcePoint; // source is same
    float2 wrappedNextMid = (wrappedNextTermPos + sourcePoint) * 0.5f;
    float wrappedNextLen = length(wrappedNextDelta) + 1e-6;
    float2 wrappedNextPerp = float2(-wrappedNextDelta.y, wrappedNextDelta.x) / wrappedNextLen;
    float2 wrappedNextJoint = wrappedNextPerp * width;
    
    // select between SIMD and wrapped values based on seam detection
    float2 prevDelta = select(simdPrevDelta, wrappedPrevDelta, isUpSeam);
    float2 nextDelta = select(simdNextDelta, wrappedNextDelta, isDownSeam);
    float2 prevMidpoint = select(simdPrevMidpoint, wrappedPrevMid, isUpSeam);
    float2 nextMidpoint = select(simdNextMidpoint, wrappedNextMid, isDownSeam);
    float2 prevJoint = select(simdPrevJoint, wrappedPrevJoint, isUpSeam);
    float2 nextJoint = select(simdNextJoint, wrappedNextJoint, isDownSeam);
    
    // prev miter joint
    float2 prevP1 = prevMidpoint + prevJoint;
    float2 prevP2 = midpoint - joint;
    
    float2 prevD1 = prevDelta;
    float2 prevD2 = delta;
    
    float2 prevP1P2 = prevP2 - prevP1;
    float prevCrossProduct = prevD1.x * prevD2.y - prevD1.y * prevD2.x;

    float2 prevIntersection;
    float prevT = (prevP1P2.x * prevD2.y - prevP1P2.y * prevD2.x) / (prevCrossProduct + 1e-10);
    float2 prevIntersectionCalc = prevP1 + prevT * prevD1;
    float2 prevIntersectionFallback = (prevP1 + prevP2) * 0.5f;
    bool prevParallel = abs(prevCrossProduct) < 1e-5;
    prevIntersection = select(prevIntersectionCalc, prevIntersectionFallback, prevParallel);
    
    // next miter joint
    float2 nextP1 = midpoint + joint;
    float2 nextP2 = nextMidpoint - nextJoint;
    
    float2 nextD1 = delta;
    float2 nextD2 = nextDelta;
    
    float2 nextP1P2 = nextP2 - nextP1;
    float nextCrossProduct = nextD1.x * nextD2.y - nextD1.y * nextD2.x;

    float2 nextIntersection;
    float nextT = (nextP1P2.x * nextD2.y - nextP1P2.y * nextD2.x) / (nextCrossProduct + 1e-10);
    float2 nextIntersectionCalc = nextP1 + nextT * nextD1;
    float2 nextIntersectionFallback = (nextP1 + nextP2) * 0.5f;
    bool nextParallel = abs(nextCrossProduct) < 1e-5;
    nextIntersection = select(nextIntersectionCalc, nextIntersectionFallback, nextParallel);
    
    //screenspace
    float2 origin_t = (sourcePoint * transform.scale) + transform.offset;

    float2 prevIntersection_t = (prevIntersection * transform.scale) + transform.offset;
    float2 nextIntersection_t = (nextIntersection * transform.scale) + transform.offset;
    
    float2 prevJoint_t = (prevP1 * transform.scale) + transform.offset;  // prevMidpoint + prevJoint
    float2 leftJoint_t = (prevP2 * transform.scale) + transform.offset;      // midpoint - joint
    float2 rightJoint_t = (nextP1 * transform.scale) + transform.offset;      // midpoint + joint
    float2 nextJoint_t = (nextP2 * transform.scale) + transform.offset;  // nextMidpoint - nextJoint
    
    float2 prevMidpoint_t = (prevMidpoint * transform.scale) + transform.offset;
    float2 midpoint_t = (midpoint * transform.scale) + transform.offset;
    float2 nextMidpoint_t = (nextMidpoint * transform.scale) + transform.offset;

    
    payload.origin[tid] = origin_t;
    payload.midpoint[tid] = midpoint_t;
    payload.gid[tid] = gid;
    
    payload.prevIntersection[tid] = prevIntersection_t;
    payload.nextIntersection[tid] = nextIntersection_t;

    payload.prevJoint[tid] = prevJoint_t;
    payload.leftJoint[tid] = leftJoint_t;
    payload.rightJoint[tid] = rightJoint_t;
    payload.nextJoint[tid] = nextJoint_t;
    
    payload.prevMidpoint[tid] = prevMidpoint_t;
    payload.nextMidpoint[tid] = nextMidpoint_t;


    // Dispatch logic
    // Number of intermediate sample points on the bezier curve (results in numCurveLines + 1 line segments for the curve)
    // Max value depends on mesh limits: must satisfy threads_per_threadgroup * (numCurveLines + 2) <= 256
    // With 32 threads: max numCurveLines = floor(256/32) - 6 = 8
    constexpr int lod = 6; // 1 is the base case
     
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
    v.position = float4(payload.nextIntersection[tid], 0.0, 1.0);
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
    c.position = float4(payload.leftJoint[tid],        0.0, 1.0);
    d.position = float4(payload.prevIntersection[tid], 0.0, 1.0);
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
    
    constexpr int numVerticesPerThread = (256 / SIMDGROUP_SIZE);  // 4 left + 4 right
    constexpr int numLinesPerThread = numVerticesPerThread - 2;     // 3 left + 3 right
    constexpr int numPointsPerSide = numVerticesPerThread / 2;
    
    if (tid == 0) {
        output.set_primitive_count(count * numLinesPerThread);
    }
    
    uint primBase = tid * numLinesPerThread;
    uint vertBase = tid * numVerticesPerThread;
    
    float hue = float(payload.gid[tid]) / float(numEdges);
    
    PrimOut p;
    p.color = colorConvert(hue, 0.8, 1.0);
    
    uint leftBezierIndices[numPointsPerSide];
    uint rightBezierIndices[numPointsPerSide];
    
    VertexOut leftBezierVertices[numPointsPerSide];
    VertexOut rightBezierVertices[numPointsPerSide];
    
    float2 leftJoint = payload.leftJoint[tid];
    float2 rightJoint = payload.rightJoint[tid];
    
    float2 prevIntersection = payload.prevIntersection[tid];
    float2 nextIntersection = payload.nextIntersection[tid];
    
    float2 prevJoint = payload.prevJoint[tid];
    float2 nextJoint = payload.nextJoint[tid];
    
    int numLinesPerSide = numPointsPerSide - 1;
    int totalPoints = 1 + (int)payload.lod * numLinesPerSide;
    int startIdx = (int)bid * numLinesPerSide;
    
    // left side vertices (indices 0-3)
    for (int i = 0; i < numPointsPerSide; i++) {
        int globalIdx = startIdx + i;
        float t = (float)globalIdx / (float)(totalPoints - 1);
        float2 leftCurvePoint = conicBezier(t, leftJoint, prevIntersection, prevJoint, 3.0f);
        leftBezierIndices[i] = vertBase + i;
        leftBezierVertices[i].position = float4(leftCurvePoint, 0.0, 1.0);
        leftBezierVertices[i].screenPosition = leftCurvePoint;
        output.set_vertex(leftBezierIndices[i], leftBezierVertices[i]);
    }
    
    //right side vertices (indices 4-7)
    for (int i = 0; i < numPointsPerSide; i++) {
        int globalIdx = startIdx + i;
        float t = (float)globalIdx / (float)(totalPoints - 1);
        float2 rightCurvePoint = conicBezier(t, rightJoint, nextIntersection, nextJoint, 3.0f);
        rightBezierIndices[i] = vertBase + numPointsPerSide + i;
        rightBezierVertices[i].position = float4(rightCurvePoint, 0.0, 1.0);
        rightBezierVertices[i].screenPosition = rightCurvePoint;
        output.set_vertex(rightBezierIndices[i], rightBezierVertices[i]);
    }

    // left side lines: 0 -> 1, 1 -> 2, 2 -> 3
    for (int i = 0; i < numLinesPerSide; ++i) {
        output.set_primitive(primBase + i, p);
        output.set_index((primBase + i) * 2 + 0, leftBezierIndices[i]);
        output.set_index((primBase + i) * 2 + 1, leftBezierIndices[i + 1]);
    }
    
    // right side lines: 4 -> 5, 5 -> 6, 6 -> 7
    for (int i = 0; i < numLinesPerSide; ++i) {
        output.set_primitive(primBase + numLinesPerSide + i, p);
        output.set_index((primBase + numLinesPerSide + i) * 2 + 0, rightBezierIndices[i]);
        output.set_index((primBase + numLinesPerSide + i) * 2 + 1, rightBezierIndices[i + 1]);
    }
}

[[mesh]] void meshCurveTriangleShader( TriangleMeshType output,
                                       const object_data Payload& payload [[payload]],
                                       ushort tid [[thread_index_in_threadgroup]],
                                       uint bid [[threadgroup_position_in_grid]],
                                       ushort lane_id [[thread_index_in_simdgroup]],
                                       ushort threads_per_threadgroup [[threads_per_threadgroup]],
                                       ushort threads_per_simdgroup [[threads_per_simdgroup]]
) {
    uint count = payload.numEdges;
    
    constexpr int numVerticesPerThread = (256 / SIMDGROUP_SIZE);
    constexpr int numPointsPerSide = numVerticesPerThread / 2;
    constexpr int numSegments = numPointsPerSide - 1;
    constexpr int numTrianglesPerThread = numSegments * 2;
    
    if (tid == 0) {
        output.set_primitive_count(count * numTrianglesPerThread);
    }
    
    uint primBase = tid * numTrianglesPerThread;
    uint vertBase = tid * numVerticesPerThread;
    
    uint leftBezierIndices[numPointsPerSide];
    uint rightBezierIndices[numPointsPerSide];
    
    VertexOut leftBezierVertices[numPointsPerSide];
    VertexOut rightBezierVertices[numPointsPerSide];
    
    float tValues[numPointsPerSide];
    
    float2 leftJoint = payload.leftJoint[tid];
    float2 rightJoint = payload.rightJoint[tid];
    
    float2 prevIntersection = payload.prevIntersection[tid];
    float2 nextIntersection = payload.nextIntersection[tid];
    
    float2 prevJoint = payload.prevJoint[tid];
    float2 nextJoint = payload.nextJoint[tid];
    
    int totalPoints = 1 + (int)payload.lod * numSegments;
    int startIdx = (int)bid * numSegments;
    
    // left side vertices (indices 0-3)
    for (int i = 0; i < numPointsPerSide; i++) {
        int globalIdx = startIdx + i;
        float t = (float)globalIdx / (float)(totalPoints - 1);
        tValues[i] = t / 2.0;  // Store the t-value for this vertex
        
        float2 leftCurvePoint = conicBezier(tValues[i], leftJoint, prevIntersection, prevJoint, 3.0f);
        leftBezierIndices[i] = vertBase + i;
        leftBezierVertices[i].position = float4(leftCurvePoint, 0.0, 1.0);
        leftBezierVertices[i].screenPosition = leftCurvePoint;
        output.set_vertex(leftBezierIndices[i], leftBezierVertices[i]);
    }
    
    // right side vertices (indices 4-7)
    for (int i = 0; i < numPointsPerSide; i++) {
        int globalIdx = startIdx + i;
        float t = (float)globalIdx / (float)(totalPoints - 1);
        float tVal = t / 2.0;
        
        float2 rightCurvePoint = conicBezier(tVal, rightJoint, nextIntersection, nextJoint, 3.0f);
        rightBezierIndices[i] = vertBase + numPointsPerSide + i;
        rightBezierVertices[i].position = float4(rightCurvePoint, 0.0, 1.0);
        rightBezierVertices[i].screenPosition = rightCurvePoint;
        output.set_vertex(rightBezierIndices[i], rightBezierVertices[i]);
    }

    float hue = float(payload.gid[tid]) / float(numEdges);
    
    // there is a bridge between the left and right sides of the curve
    // each segment is 2 triangles
    for (int i = 0; i < numSegments; ++i) {
        uint triIdx = primBase + (i * 2);
        
        output.set_vertex(leftBezierIndices[i], leftBezierVertices[i]);
        output.set_vertex(leftBezierIndices[i + 1], leftBezierVertices[i + 1]);
        output.set_vertex(rightBezierIndices[i], rightBezierVertices[i]);
        
        // triangle: (left[i], left[i+1], right[i])
        PrimOut p1;
        p1.color = colorConvert(hue, 0.8, 1.0);
        p1.leftJoint = leftJoint;
        p1.prevIntersection = prevIntersection;
        p1.prevJoint = prevJoint;
        p1.rightJoint = rightJoint;
        p1.nextIntersection = nextIntersection;
        p1.nextJoint = nextJoint;
        p1.weight = 3.0f;
        
        p1.t0 = tValues[i];      // left[i]
        p1.t1 = tValues[i + 1];  // left[i+1]
        p1.t2 = tValues[i];      // right[i] - same t as left[i]
        
        output.set_primitive(triIdx, p1);
        output.set_index(triIdx * 3 + 0, leftBezierIndices[i]);
        output.set_index(triIdx * 3 + 1, leftBezierIndices[i + 1]);
        output.set_index(triIdx * 3 + 2, rightBezierIndices[i]);
        
        output.set_vertex(leftBezierIndices[i + 1], leftBezierVertices[i + 1]);
        output.set_vertex(rightBezierIndices[i + 1], rightBezierVertices[i + 1]);
        output.set_vertex(rightBezierIndices[i], rightBezierVertices[i]);
        
        
        // triangle: (left[i+1], right[i+1], right[i])
        PrimOut p2;
        p2.color = colorConvert(hue, 0.8, 1.0);
        p2.leftJoint = leftJoint;
        p2.prevIntersection = prevIntersection;
        p2.prevJoint = prevJoint;
        p2.rightJoint = rightJoint;
        p2.nextIntersection = nextIntersection;
        p2.nextJoint = nextJoint;
        p2.weight = 3.0f;
        
        p2.t0 = tValues[i + 1];  // left[i+1]
        p2.t1 = tValues[i + 1];  // right[i+1] - same t
        p2.t2 = tValues[i];      // right[i]
        
        output.set_primitive(triIdx + 1, p2);
        output.set_index((triIdx + 1) * 3 + 0, leftBezierIndices[i + 1]);
        output.set_index((triIdx + 1) * 3 + 1, rightBezierIndices[i + 1]);
        output.set_index((triIdx + 1) * 3 + 2, rightBezierIndices[i]);
    }
}

fragment float4 fragmentSimple(FragmentIn in [[stage_in]], float3 barycentricCoord [[barycentric_coord]]) {
    return float4(in.p.color, 1.0);
}

fragment float4 fragmentBody(FragmentIn in [[stage_in]],
                             float3 barycentricCoord [[barycentric_coord]] ) {
    float2 fragPos = in.v.screenPosition;

    float t = barycentricCoord.x * in.p.t0 + barycentricCoord.y * in.p.t1 + barycentricCoord.z * in.p.t2;
    
    float2 leftPoint = conicBezier(t, in.p.leftJoint, in.p.prevIntersection, in.p.prevJoint, in.p.weight);
    float2 rightPoint = conicBezier(t, in.p.rightJoint, in.p.nextIntersection, in.p.nextJoint, in.p.weight);
    
    float distToLeft = distance(fragPos, leftPoint);
    float distToRight = distance(fragPos, rightPoint);
    
    float edgeDist = min(distToLeft, distToRight);
        float strokeWidth = distance(leftPoint, rightPoint);
    
    float normalizedDist = edgeDist / (strokeWidth * 0.5);
    
    float intensity = 1.0 - saturate(normalizedDist);
    //float intensity = saturate(1.0 - abs(normalizedDist - 0.9) * 10.0);
    
    return float4(intensity * in.p.color, 1.0);
}


#endif


