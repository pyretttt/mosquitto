#pragma once

#include "Eigen/Dense"
#include "SDL.h"

#include "Renderer.h"
#include "SDLController.h"
#include "SDLRenderer.h"

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

        SDLRenderer::MeshData meshData = {
            Mesh(
                {
                    Eigen::Vector3f(100, 300, 1),
                    Eigen::Vector3f(300, 300, 1),
                    Eigen::Vector3f(200, 100, 1),
                },
                {
                    Face{0, 1, 2, {}}
                }
            )
        };
        while (!shouldClose) {
            auto dt = SDL_GetTicks() - previousFrameTicks;
            if (dt < frameTime) {
                SDL_Delay(frameTime - dt);
                dt = SDL_GetTicks() - previousFrameTicks;
            }

            processInput();
            sdlController.renderer->update(meshData, dt);
            sdlController.renderer->render();
            previousFrameTicks = SDL_GetTicks();
        }
    }

private:
    GameLoop() : sdlController(SDLController(RendererType::CPU, {800, 600})) {
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
                case SDLK_ESCAPE:
                    shouldClose = true;
                    SDL_Quit();
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

    SDLController sdlController;
    bool shouldClose = false;
    int previousFrameTicks = 0;
    const int frameRate = 60;
    const int frameTime = 1000 / frameRate;
};