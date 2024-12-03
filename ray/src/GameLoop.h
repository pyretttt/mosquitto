#pragma once

#include "SDL.h"
#include "Eigen/Dense"

#include "SDLController.h"
#include "SDLRenderer.h"
#include "Renderer.h"

class GameLoop {
public:
    GameLoop(GameLoop const &other) = delete;
    GameLoop &operator=(GameLoop const &other) = delete;

    inline static GameLoop &instance() {
        static GameLoop loop;
        return loop;
    }

    void start() {
        sdlController.showWindow();
        while (!shouldClose) {
            processInput();

            Eigen::Vector2f a, b;
            a(0, 0) = 100;
            a(1, 0) = 100;
            b(0, 0) = 300;
            b(1, 0) = 300;


        }

        sdlController.~SDLController();
    }

private:
    GameLoop() : sdlController(SDLController({800, 600})) {
        // renderer = std::make_unique(SDLRenderer(sdlController));
    }

    inline void processInput() {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
            case SDL_KEYDOWN:
                switch (event.key.keysym.sym) {
                case SDLK_UP:
                    break;
                case SDLK_DOWN:
                    break;
                }
                break;
            case SDL_QUIT:
            case SDL_WINDOWEVENT_CLOSE:
                shouldClose = true;
                break;
            default:
                break;
            }
        }
    }

    std::unique_ptr<Renderer> renderer;
    SDLController sdlController;
    bool shouldClose = false;
};