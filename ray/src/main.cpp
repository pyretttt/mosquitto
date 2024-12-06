#include <iostream>
#include <memory>

#include "SDL.h"

#include "GameLoop.cpp"
#include "SDLController.h"

int main() {
    if (SDL_Init(SDL_INIT_EVERYTHING)) {
        std::cerr << "Failed to initialize SDL: " << SDL_GetError() << std::endl;
        return -1;
    }
    std::cout << eye<3>() << std::endl;
    GameLoop::instance().start();
    return 0;
}