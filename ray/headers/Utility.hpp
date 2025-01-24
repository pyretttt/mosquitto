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

    return a | (r & 0xFF000000) | (g & 0x00FF0000) | (b & 0x0000FF00);
}

inline constexpr uint32_t interpolateRGBAColor(
    uint32_t first,
    uint32_t second,
    float p
) {
    // TODO: Improve calculations somehow
    float alpha = std::min(std::max(0.f, p), 1.0f);

    uint32_t r = 0xFF000000 & (uint32_t)((first & 0xFF000000) * alpha + (second & 0xFF000000) * (1 - alpha));
    uint32_t g = 0x00FF0000 & (uint32_t)((first & 0x00FF0000) * alpha + (second & 0x00FF0000) * (1 - alpha));
    uint32_t b = 0x0000FF00 & (uint32_t)((first & 0x0000FF00) * alpha + (second & 0x0000FF00) * (1 - alpha));
    uint32_t a = 0x000000FF & (uint32_t)((first & 0x000000FF) * alpha + (second & 0x000000FF) * (1 - alpha));

    return r | g | b | a;
}

constexpr inline bool isApproximatelyEqual(float value, float test, float delta = 1e-8f) {
    return ((test - delta) < value < (test + delta));
}