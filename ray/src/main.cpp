#include <iostream>

#include "SDL.h"

#include "WindowController.h"

WindowController windowController;

int main() {
    if (!SDL_Init(SDL_INIT_EVERYTHING)) {
        std::cerr << "Failed to initialize SDL" << std::endl;
        return -1;
    }

    windowController = WindowController(std::make_pair(800, 600));
    windowController.showWindow();
    return 0;
}