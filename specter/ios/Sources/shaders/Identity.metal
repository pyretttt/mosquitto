#include <metal_stdlib>
#include <SwiftUI/SwiftUI.h>
using namespace metal;

float parametrizedSin(
    float x,
    float freq,
    float phase = 0.0,
    float magnitude = 1.0,
    float shift = 0.0
) {
    return magnitude * sin(freq * x + phase) + shift;
}

[[ stitchable ]]
half4 identity(float2 position, half4 color, float4 frame, float time) {
    float2 localCoordinates = float2((position.x - frame.x) / frame.z, (position.y - frame.y) / frame.w);
    float t = sin(time) * 0.5 + 0.5;
    half4 c = half4(
        parametrizedSin(time + localCoordinates.x, 1.5, 0.0, 0.3, 0.3),
        parametrizedSin(time + localCoordinates.y, 2.0, M_PI_F, 0.5, 0.5),
        parametrizedSin(time + localCoordinates.y / localCoordinates.x, 7.0, M_PI_F / 2.0, 0.1, 0.5),
        1.0
    );
    return mix(c, half4(0.3, 0.3, 0.3, 1.0), 0.7);
}
