#include <iostream>
#include <memory>

#include "Eigen/Dense"
#include "SDL.h"

#include "GameLoop.cpp"
#include "MathUtils.h"
#include "Renderer.h"
#include "SDLController.h"
#include "Utility.h"

int main() {
    if (SDL_Init(SDL_INIT_EVERYTHING)) {
        std::cerr << "Failed to initialize SDL: " << SDL_GetError() << std::endl;
        return -1;
    }
    GameLoop::instance().start();
    return 0;
}