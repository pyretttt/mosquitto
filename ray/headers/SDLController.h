#pragma once

#include <functional>
#include <utility>

#include "SDL.h"

#include "Renderer.h"


struct SDLController {
    SDLController(RendererType renderType, std::pair<int, int> windowSize);
    ~SDLController();

    void showWindow();

    std::shared_ptr<Renderer> renderer;
private:
    std::function<void()> windowInit;
    SDL_Window *window;
    RendererType rendererType;
};