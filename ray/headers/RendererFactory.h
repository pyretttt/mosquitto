#pragma once

#include "SDL.h"

#include "GlobalConfig.h"
#include "Lazy.h"
#include "ReactivePrimitives.h"
#include "RendererBase.h"
#include "sdl/Camera.h"
#include "sdl/Renderer.h"

struct RenderFactoryParams {
    RenderFactoryParams(GlobalConfig const &globalConfig) : globalConfig(globalConfig) {
    };

    SDL_Window *window = nullptr;
    GlobalConfig globalConfig;
};

std::shared_ptr<Renderer> inline makeRenderer(RenderFactoryParams const &params) {
    switch (params.globalConfig.rendererType.value()) {
    case RendererType::CPU:
        return std::make_shared<sdl::Renderer>(
            params.window,
            params.globalConfig.windowSize.value(),
            Lazy<sdl::Camera>([config = params.globalConfig]() {
                return sdl::Camera(config.fov, config.windowSize);
            })
        );
    case RendererType::OpenGL:
        std::exit(1);
    }
}