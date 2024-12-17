#pragma once

#include <functional>
#include <utility>

#include "SDL.h"

#include "RendererBase.h"
#include "GlobalConfig.h"

struct SDLController {
    SDLController(
        GlobalConfig config
    );
    ~SDLController();

    void showWindow();

    std::shared_ptr<Renderer> renderer;
private:
    std::function<void()> windowInit;
    SDL_Window *window;
    GlobalConfig config;
};