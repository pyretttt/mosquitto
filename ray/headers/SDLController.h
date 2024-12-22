#pragma once

#include <functional>
#include <utility>

#include "SDL.h"

#include "RendererBase.h"
#include "GlobalConfig.h"

struct SDLController {
    SDLController() = delete;
    SDLController(SDLController const &other) = delete;
    SDLController(SDLController &&other) = delete;
    SDLController operator=(SDLController const &other) = delete;
    SDLController operator=(SDLController &&other) = delete;
    explicit SDLController(
        GlobalConfig const &config
    );
    ~SDLController();

    void showWindow();

    std::shared_ptr<Renderer> renderer;
private:
    std::function<void()> windowInit;
    SDL_Window *window;
    GlobalConfig const &config;
};