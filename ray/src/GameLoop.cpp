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
            MeshBuffer{
                {
                    ml::Vector3f(-0.5, 0.5, 0.5),   // 0. left - top - near
                    ml::Vector3f(-0.5, -0.5, 0.5),  // 1. left - bottom - near
                    ml::Vector3f(0.5, 0.5, 0.5),    // 2. right - top - near
                    ml::Vector3f(0.5, -0.5, 0.5),   // 3. right - bottom - near
                    ml::Vector3f(-0.5, 0.5, -0.5),  // 4. left - top - far
                    ml::Vector3f(-0.5, -0.5, -0.5), // 5. left - bottom - far
                    ml::Vector3f(0.5, 0.5, -0.5),   // 6. right - top - far
                    ml::Vector3f(0.5, -0.5, -0.5),   // 7. right - bottom - far
                },
                {
                    // Face{7, 5, 6, {}}, // far front
                    // Face{4, 6, 5, {}},

                    // Face{1, 2, 0, {}}, // near front
                    // Face{1, 3, 2, {}},

                    Face{0, 2, 6, {}}, // top
                    Face{0, 6, 4, {}},

                    // Face{3, 1, 7, {}}, // bottom
                    // Face{7, 5, 3, {}},

                    // Face{5, 1, 4, {}}, // left
                    // Face{0, 4, 1, {}},

                    // Face{3, 7, 2, {}}, // right
                    // Face{6, 2, 7, {}},
                }
            }
        };

        while (!shouldClose) {
            auto currentTicks = SDL_GetTicks();
            auto dt = currentTicks - previousFrameTicks;
            if (dt < frameTime) {
                SDL_Delay(frameTime - dt);
                dt = SDL_GetTicks() - previousFrameTicks;
            }

            ml::Matrix4f transformationMatrix = ml::scaleMatrix(10, 10, 10);
            transformationMatrix = ml::matMul(
                ml::rodriguezRotationMatrix({0, 1, 0}, static_cast<float>(currentTicks) / 1000),
                transformationMatrix
            );
            transformationMatrix = ml::matMul(ml::translationMatrix(0, -20, -50), transformationMatrix);
            node[0].transform = transformationMatrix;

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