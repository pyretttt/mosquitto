#pragma once

#include "SDL.h"

void inline destructSDLWindow(SDL_Window *window) noexcept {
    SDL_DestroyWindow(window);
}