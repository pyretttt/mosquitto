#include <iostream>

#include "Eigen/Dense"
#include "SDL.h"
#include "rpp/rpp.hpp"

#include "MathUtils.h"
#include "Renderer.h"
#include "SDLController.h"

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
        Renderer::MeshData node{
            MeshBuffer(
                {
                    Vector3f(-40, 25, 100),
                    Vector3f(25, 25, 100),
                    Vector3f(0, -25, 100),
                },
                {
                    Face{0, 1, 2, {}},
                }
            )
        };
        while (!shouldClose) {
            auto currentTicks = SDL_GetTicks();
            auto dt = currentTicks - previousFrameTicks;
            if (dt < frameTime) {
                SDL_Delay(frameTime - dt);
                dt = SDL_GetTicks() - previousFrameTicks;
            }

            auto rotationMatrix = rodriguezRotationMatrix({0, 0, 1}, static_cast<float>(dt) / 10000);
            node[0].transform = matMul(rotationMatrix, node[0].transform);

            processInput();
            sdlController.renderer->update(node, dt);
            sdlController.renderer->render();
            previousFrameTicks = SDL_GetTicks();
        }
    }

private:
    GameLoop() : windowSize({800, 600}), sdlController(SDLController(RendererType::CPU, windowSize.get_value())) {
        windowSize.get_observable()
            .subscribe(
                [](std::pair<int, int> screenSize) {
                    std::cout << "New screen size " << screenSize.first << " x " << screenSize.second << std::endl;
                },
                [](std::exception_ptr err) {},
                []() {}
            );
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
            case SDL_WINDOWEVENT:
                switch (event.window.event) {
                case SDL_WINDOWEVENT_RESIZED:
                    windowSize.get_observer().on_next({event.window.data1, event.window.data2});
                }
                break;
            default:
                break;
            }
        }
    }

    rpp::subjects::behavior_subject<std::pair<int, int>> windowSize;
    SDLController sdlController;
    bool shouldClose = false;
    int previousFrameTicks = 0;
    const int frameRate = 60;
    const int frameTime = 1000 / frameRate;
};