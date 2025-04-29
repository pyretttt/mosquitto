#pragma once

#include "SDL.h"

#include "GlobalConfig.hpp"
#include "Lazy.hpp"
#include "ReactivePrimitives.hpp"
#include "RendererBase.hpp"
#include "sdl/Camera.hpp"
#include "sdl/Renderer.hpp"

struct RenderFactoryParams {
    explicit RenderFactoryParams(std::shared_ptr<GlobalConfig> globalConfig)
     : globalConfig(globalConfig) {};

    std::shared_ptr<GlobalConfig> globalConfig;
};

std::shared_ptr<Renderer> inline makeRenderer(
    RenderFactoryParams const &params
) {
    switch (params.globalConfig->rendererType.value()) {
    case RendererType::CPU:
        return std::make_shared<sdl::Renderer>(
            params.window,
            params.globalConfig.windowSize.value(),
            Lazy<sdl::Camera>([&config = params.globalConfig]() {
                return std::make_shared<sdl::Camera>(config.fov.asObservableObject(), config.windowSize.asObservableObject());
            })
        );
    case RendererType::OpenGL:
        std::exit(1);
    }
}