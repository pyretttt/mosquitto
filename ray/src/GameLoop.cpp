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
            processInput();

            sdlController.renderer->update(meshData);
            sdlController.renderer->render();
            std::cout << "Loop iteration" << std::endl;
        }

        sdlController.~SDLController();
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