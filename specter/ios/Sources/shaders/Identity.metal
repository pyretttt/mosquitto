#include <metal_stdlib>
#include <SwiftUI/SwiftUI.h>
using namespace metal;


[[ stitchable ]]
half4 identity(float2 position, half4 color) {
    return half4(0.5, 0.0, 0.0, 1.0);
}
