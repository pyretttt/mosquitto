#include <exception>
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
    ObservableProperty<float> fov(0.89);
    ObservableProperty<std::pair<size_t, size_t>> windowSize({800, 600});

    auto camera = sdl::Camera(fov, windowSize);
    windowSize.value({10, 10});
    ml::Matrix4f m0 = camera.getScenePerspectiveProjectionMatrix();
    windowSize.value({20, 20});
    ml::Matrix4f m1 = camera.getScenePerspectiveProjectionMatrix();

    RunLoop::instance()
        .start();
    return 0;
}