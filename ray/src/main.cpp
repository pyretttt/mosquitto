#include <iostream>

#include "SDL.h"

#include "WindowController.h"
#include "math/Vector3.h"

WindowController windowController{{800, 600}};

int main() {
    if (!SDL_Init(SDL_INIT_EVERYTHING)) {
        std::cerr << "Failed to initialize SDL" << std::endl;
        return -1;
    }

    windowController.showWindow();
    return 0;
}