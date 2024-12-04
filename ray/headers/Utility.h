#pragma once
#include <iostream>
#include <cmath>
#include <functional>

template <typename T>
constexpr inline T &&modified(T &&object, std::function<void(T &)> modification) {
    modification(object);
    return object;
}

inline constexpr uint32_t interpolateColorIntensity(uint32_t argb, float intensity) {
    auto i = std::min(std::max(0.f, intensity), 1.f);
    uint32_t a = argb & 0xFF000000;
    uint32_t r = (argb & 0x00FF0000) * i;
    uint32_t g = (argb & 0x0000FF00) * i;
    uint32_t b = (argb & 0x000000FF) * i;

    uint32_t res = a | (r & 0x00FF0000) | (g & 0x0000FF00) | (b & 0x000000FF);
    return res;
}