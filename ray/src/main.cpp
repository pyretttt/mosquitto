#include <exception>
#include <iostream>
#include <memory>

#include "SDL.h"

#include "Lazy.hpp"
#include "ReactivePrimitives.hpp"
#include "RunLoop.cpp"
#include "Controller.hpp"
#include "opengl/Renderer.hpp"
#include "sdl/Camera.hpp"
#include "Plane.hpp"

int main() {
    if (SDL_Init(SDL_INIT_EVERYTHING)) {
        std::cerr << "Failed to initialize SDL: " << SDL_GetError() << std::endl;
        return -1;
    }

    ml::Vector3f a = {0, 0, 0};
    ml::Vector3f b = {10, 15, 20};

    RunLoop::instance()
        .start();
    return 0;
}