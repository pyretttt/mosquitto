#pragma once

#include <utility>
#include <memory>

#include "SDL.h"

#include "sdlUtils.hpp"
#include "ReactivePrimitives.hpp"

enum class RendererType;

struct GlobalConfig final {
    using WindowPtr = std::unique_ptr<SDL_Window, std::function<void (SDL_Window *)>>;

    GlobalConfig() = delete;

    GlobalConfig(
        ObservableProperty<RendererType> rendererType,
        ObservableProperty<std::pair<size_t, size_t>> windowSize,
        ObservableProperty<float> fov,
        ObservableProperty<float> cameraSpeed,
        ObservableProperty<float> mouseSpeed
    ) 
        : rendererType(std::move(rendererType))
        , windowSize(std::move(windowSize))
        , fov(std::move(fov))
        , cameraSpeed(std::move(cameraSpeed))
        , mouseSpeed(std::move(mouseSpeed)) {}

    GlobalConfig(
        GlobalConfig &&other
    )
        : rendererType(std::move(other.rendererType))
        , windowSize(std::move(other.windowSize))
        , fov(std::move(other.fov))
        , cameraSpeed(std::move(other.cameraSpeed))
        , mouseSpeed(std::move(other.mouseSpeed)) {}

    GlobalConfig(GlobalConfig const &other) = delete;
    GlobalConfig operator=(GlobalConfig const &other) = delete;
    GlobalConfig operator=(GlobalConfig &&other) = delete;

    ObservableProperty<RendererType> rendererType;
    ObservableProperty<std::pair<size_t, size_t>> windowSize;
    ObservableProperty<float> fov;
    ObservableProperty<float> cameraSpeed;
    ObservableProperty<float> mouseSpeed;
    WindowPtr window = WindowPtr(nullptr, destructSDLWindow);
};