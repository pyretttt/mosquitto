#include <memory>

#include "SDL.h"

#include "Core.hpp"
#include "Errors.hpp"
#include "ReactivePrimitives.hpp"
#include "RendererFactory.hpp"
#include "SDLController.hpp"
#include "Utility.hpp"
#include "sdl/Renderer.hpp"

SDLController::SDLController(
    GlobalConfig const &config
) : config(config) {
    windowInit = [&config, this]() {
        auto winSize{config.windowSize.value()};
        this->window = SDL_CreateWindow(
            nullptr,
            SDL_WINDOWPOS_CENTERED,
            SDL_WINDOWPOS_CENTERED,
            winSize.first,
            winSize.second,
            SDL_WINDOW_BORDERLESS | SDL_WINDOW_RESIZABLE
        );
    };
}

SDLController::~SDLController() { SDL_DestroyWindow(window); }

void SDLController::showWindow() {
    windowInit();
    if (!window) {
        throw Errors::WindowInitFailed;
    }
    renderer = makeRenderer(
        modified<RenderFactoryParams>(RenderFactoryParams(config), [this](RenderFactoryParams &params) {
            params.window = this->window;
        })
    );
}
