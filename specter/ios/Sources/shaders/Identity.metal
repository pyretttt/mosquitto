#include <metal_stdlib>
#include <SwiftUI/SwiftUI.h>
using namespace metal;

// Pseudo random determenistic over `p`
float hash21(float2 p) {
    p = fract(p * float2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

// Noise deterministic over `p`, slight change of `p` will result in completely different result
float noise(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    // Compute pseudo random for four corners of 2x2 grid
    float a = hash21(i);
    float b = hash21(i + float2(1.0, 0.0));
    float c = hash21(i + float2(0.0, 1.0));
    float d = hash21(i + float2(1.0, 1.0));
    float2 u = f * f * (3.0 - 2.0 * f); // Non linear interpolation for [0, 1]
    return mix(a, b, u.x) // interpolate horizontally between top edges (a, b)
        + (c - a) * u.y * (1.0 - u.x) // adds vertical influence of left edge (c, a) interpolated by `u.y`
        + (d - b) * u.y * u.x; // adds vertical influces of right edge (d, b) interpolated by `u.y`
}

// fractal brownian motion
float fbm(float2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    float2 shift = float2(100.0, 100.0);
    for (int i = 0; i < 5; i++) {
        value += amplitude * noise(p);
        p = p * 2.0 + shift;
        amplitude *= 0.5;
    }
    return value;
}

float2 rotate(float2 p, float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return float2(c * p.x - s * p.y, s * p.x + c * p.y);
}

[[ stitchable ]]
half4 identity(float2 position, half4 color, float4 frame, float time) {
    float2 uv = (position - frame.xy) / frame.zw;
    float aspect = frame.z / frame.w;

    float2 p = (uv - 0.5) * 2.0 * float2(aspect, 1.0);
    float t = time;

    float2 q = rotate(p, sin(t * 0.35) * M_PI_F); // Rotates fragment
    float2 flow = float2(
        sin(q.y * 1.6 + t * 2.1) + 0.5 * sin(q.y * 3.2 - t * 1.1),
        cos(q.x * 1.4 - t * 1.7) + 0.5 * cos(q.x * 2.8 + t * 0.9)
    );
    q += 0.12 * flow;

    float n = fbm(q * 1.8 + float2(t * 0.6, -t * 0.4));
    float m = fbm(q * 3.6 - float2(t * 0.4, t * 0.2));
    float motion = sin((q.x + q.y) * 2.2 + t * 2.3) * 0.5 + 0.5;

    float diag = q.x * 0.35 + q.y * 0.55;
    float g1 = smoothstep(-0.7, 0.8, diag + n * 0.6);
    float g2 = smoothstep(-0.2, 1.0, (q.y * 0.6 - q.x * 0.25) + m * 0.4);

    // Colors
    float3 c1 = float3(0.87, 0.79, 1.0);
    float3 c2 = float3(0.57, 0.40, 0.98);
    float3 c3 = float3(0.16, 0.09, 0.28);
    float3 c4 = float3(0.73, 0.52, 0.98);

    float3 base = mix(c1, c2, g1);
    base = mix(base, c4, g2 * (0.5 + 0.35 * motion));

    float pool = smoothstep(0.2, 0.85, length(q - float2(0.6, -0.2)) + n * 0.2);
    base = mix(base, c3, (1.0 - pool) * 0.85);

    float vignette = smoothstep(1.25, 0.5, length(p));
    base = mix(c3, base, vignette);

    return half4(half3(base), 1.0);
}
