#include <iostream>
#include <memory>

#include "SDL.h"

#include "RunLoop.cpp"
#include "SDLController.h"

int main() {
    if (SDL_Init(SDL_INIT_EVERYTHING)) {
        std::cerr << "Failed to initialize SDL: " << SDL_GetError() << std::endl;
        return -1;
    }
    std::cout << ml::eye<4>() << std::endl;
    RunLoop::instance().start();
    return 0;
}