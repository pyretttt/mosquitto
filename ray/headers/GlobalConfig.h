#pragma once

#include <utility>

#include "ReactivePrimitives.h"

enum class RendererType;

struct GlobalConfig final {
    GlobalConfig() = delete;

    GlobalConfig(
        ObservableProperty<RendererType> rendererType,
        ObservableProperty<std::pair<size_t, size_t>> windowSize,
        ObservableProperty<float> fov
    ) : rendererType(rendererType), windowSize(windowSize), fov(fov) {}


    GlobalConfig(GlobalConfig const &other) : rendererType(other.rendererType), windowSize(other.windowSize), fov(other.fov) {
    }
    GlobalConfig(GlobalConfig &&other) : rendererType(std::move(other.rendererType)), windowSize(std::move(other.windowSize)), fov(std::move(other.fov)) {
    };

    GlobalConfig operator=(GlobalConfig const &other) = delete;
    GlobalConfig operator=(GlobalConfig &&other) = delete;

    ObservableObject<RendererType> rendererType;
    ObservableObject<std::pair<size_t, size_t>> windowSize;
    ObservableObject<float> fov;
};