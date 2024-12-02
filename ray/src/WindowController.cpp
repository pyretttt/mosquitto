#include "WindowController.h"
#include "Errors.h"

#include "SDL.h"

WindowController::WindowController(std::pair<int, int> size) {
    windowInit = [size, this]() {
        this->window = SDL_CreateWindow(
            nullptr,
            SDL_WINDOWPOS_CENTERED,
            SDL_WINDOWPOS_CENTERED,
            size.first,
            size.second,
            SDL_WINDOW_BORDERLESS | SDL_WINDOW_RESIZABLE
        );
    };
}

WindowController::~WindowController() {
    SDL_DestroyWindow(window);
}

void WindowController::showWindow() const {
    windowInit();
    if (!window) {
        throw Errors::WindowInitFailed;
    }
}
