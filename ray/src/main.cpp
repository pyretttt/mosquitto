#include <exception>
#include <iostream>
#include <memory>

#include "SDL.h"

#include "Lazy.hpp"
#include "ReactivePrimitives.hpp"
#include "RunLoop.cpp"
#include "SDLController.hpp"
#include "sdl/Camera.hpp"

int main() {
    if (SDL_Init(SDL_INIT_EVERYTHING)) {
        std::cerr << "Failed to initialize SDL: " << SDL_GetError() << std::endl;
        return -1;
    }
    // ObservableProperty<float> fov(0.89);
    // ObservableProperty<std::pair<size_t, size_t>> windowSize({800, 600});

    // auto camera = sdl::Camera(fov, windowSize);
    // auto camera2 = camera;
    RunLoop::instance()
        .start();
    return 0;
}