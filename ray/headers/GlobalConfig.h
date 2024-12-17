#pragma once

#include <utility>

#include "ReactivePrimitives.h"

enum class RendererType;

struct GlobalConfig final {
    GlobalConfig(
        ObservableProperty<RendererType> rendererType,
        ObservableProperty<std::pair<size_t, size_t>> windowSize,
        ObservableProperty<float> fov
    ) : rendererType(rendererType), windowSize(windowSize), fov(fov) {}

    ObservableObject<RendererType> rendererType;
    ObservableObject<std::pair<size_t, size_t>> windowSize;
    ObservableObject<float> fov;
};