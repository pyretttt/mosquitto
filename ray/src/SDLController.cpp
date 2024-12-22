#include <memory>

#include "SDL.h"

#include "Errors.h"
#include "ReactivePrimitives.h"
#include "RendererFactory.h"
#include "SDLController.h"
#include "Utility.h"
#include "sdl/Renderer.h"

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
