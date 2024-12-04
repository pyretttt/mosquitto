#include <memory>

#include "SDL.h"

#include "Errors.h"
#include "RendererFactory.h"
#include "SDLController.h"
#include "sdl/SDLRenderer.h"
#include "Utility.h"

SDLController::SDLController(RendererType rendererType, std::pair<int, int> windowSize)
    : rendererType(rendererType) {
    windowInit = [windowSize, rendererType, this]() {
        auto winSize{windowSize};
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
        rendererType,
        modified<RenderFactoryParams>({}, [this](auto &params) {
            params.window = this->window;
            int w, h;
            SDL_GetWindowSize(this->window, &w, &h);
            params.resolution = {w, h};
        })
    );
}
