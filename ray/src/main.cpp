#include <iostream>
#include <memory>

#include "SDL.h"
#include "Eigen/Dense"

#include "WindowController.h"

class GameLoop {
public:
    GameLoop(GameLoop const &other) = delete;
    GameLoop &operator=(GameLoop const &other) = delete;
   
    static GameLoop &instance() {
        static GameLoop loop;
        return loop;
    }

    void start() {
        windowController.showWindow();
        for (;;);
    }
private:
    GameLoop() : windowController(WindowController({800, 600})) {}

    WindowController windowController;
};

int main() {
    if (SDL_Init(SDL_INIT_EVERYTHING)) {
        std::cerr << "Failed to initialize SDL: " << SDL_GetError() << std::endl;
        return -1;
    }

    GameLoop &instance = GameLoop::instance();
    instance.start();

    return 0;
}