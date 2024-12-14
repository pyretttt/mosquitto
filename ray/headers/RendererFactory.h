#pragma once

#include "SDL.h"

#include "RendererBase.h"
#include "sdl/Renderer.h"

struct RenderFactoryParams {
    SDL_Window *window = nullptr;
    std::pair<int, int> resolution;
};

std::shared_ptr<Renderer> inline makeRenderer(RendererType type, RenderFactoryParams params) {
    switch (type) {
    case RendererType::CPU:
        return std::make_shared<sdl::Renderer>(params.window, params.resolution);
    case RendererType::OpenGL:
        std::exit(1);
    }
}