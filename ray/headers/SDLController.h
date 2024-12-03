#pragma once

#include <functional>
#include <utility>

#include "Renderer.h"

struct SDL_Window;


struct SDLController {
    SDLController(std::pair<int, int> size);
    ~SDLController();

    void showWindow() const;
private:
    std::function<void()> windowInit;
    SDL_Window *window;
    std::shared_ptr<Renderer> renderer;
};