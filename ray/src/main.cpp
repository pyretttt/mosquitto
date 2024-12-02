#include <iostream>
#include <memory>

#include "Eigen/Dense"
#include "SDL.h"

#include "GameLoop.h"
#include "MathUtils.h"
#include "WindowController.h"

int main() {
    Eigen::Vector2f v(2.0, 3.0);
    Eigen::Vector2f y(1.0, 0.0);
    std::cout << projection(v, y) << std::endl;
    std::cout << rejection(v, y) << std::endl;

    if (SDL_Init(SDL_INIT_EVERYTHING)) {
        std::cerr << "Failed to initialize SDL: " << SDL_GetError() << std::endl;
        return -1;
    }

    GameLoop &instance = GameLoop::instance();
    instance.start();

    return 0;
}