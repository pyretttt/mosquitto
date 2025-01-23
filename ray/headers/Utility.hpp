#pragma once
#include <iostream>
#include <cmath>
#include <functional>

template <typename T>
constexpr inline T &&modified(T &&object, std::function<void(T &)> modification) {
    modification(object);
    return object;
}

inline constexpr uint32_t interpolateRGBAColorIntensity(uint32_t argb, float intensity, float minValue = 0.0f) {
    float alpha = std::min(std::max(minValue, intensity), 1.0f);
    
    uint32_t r = (argb & 0xFF000000) * alpha;
    uint32_t g = (argb & 0x00FF0000) * alpha;
    uint32_t b = (argb & 0x0000FF00) * alpha;
    uint32_t a = argb & 0x000000FF;

    uint32_t res = a | (r & 0xFF000000) | (g & 0x00FF0000) | (b & 0x0000FF00);
    return res;
}
