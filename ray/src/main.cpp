#include <iostream>
#include <memory>

#include "SDL.h"

#include "Lazy.h"
#include "ReactivePrimitives.h"
#include "RunLoop.cpp"
#include "SDLController.h"
#include "sdl/Camera.h"

int main() {
    if (SDL_Init(SDL_INIT_EVERYTHING)) {
        std::cerr << "Failed to initialize SDL: " << SDL_GetError() << std::endl;
        return -1;
    }

    RunLoop::instance().start();
    return 0;
}