#pragma once

#include <utility>

#include "ReactivePrimitives.hpp"

enum class RendererType;

struct GlobalConfig final {
    GlobalConfig() = delete;

    GlobalConfig(
        ObservableProperty<RendererType> &&rendererType,
        ObservableProperty<std::pair<size_t, size_t>> &&windowSize,
        ObservableProperty<float> &&fov
    ) : rendererType(std::move(rendererType)), windowSize(std::move(windowSize)), fov(std::move(fov)) {}

    GlobalConfig(GlobalConfig &&other) : rendererType(std::move(other.rendererType)), windowSize(std::move(other.windowSize)), fov(std::move(other.fov)) {
                                         };

    GlobalConfig(GlobalConfig const &other) = delete;
    GlobalConfig operator=(GlobalConfig const &other) = delete;
    GlobalConfig operator=(GlobalConfig &&other) = delete;

    ObservableProperty<RendererType> rendererType;
    ObservableProperty<std::pair<size_t, size_t>> windowSize;
    ObservableProperty<float> fov;
};