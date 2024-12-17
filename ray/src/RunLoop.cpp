#include <iostream>

#include "Eigen/Dense"
#include "SDL.h"

#include "GlobalConfig.h"
#include "MathUtils.h"
#include "ReactivePrimitives.h"
#include "RendererBase.h"
#include "SDLController.h"

class RunLoop {
public:
    RunLoop(RunLoop const &other) = delete;
    RunLoop &operator=(RunLoop const &other) = delete;

    inline static RunLoop &instance() {
        static RunLoop loop;
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
                    ml::Vector3f(0.5, -0.5, -0.5),  // 7. right - bottom - far
                },
                {
                    Face{5, 6, 7, {}}, // far front
                    Face{5, 4, 6, {}},

                    Face{1, 2, 0, {}}, // near front
                    Face{1, 3, 2, {}},

                    Face{0, 6, 4, {}}, // top
                    Face{0, 2, 6, {}},

                    Face{1, 7, 3, {}}, // bottom
                    Face{1, 5, 7, {}},

                    Face{1, 4, 5, {}}, // left
                    Face{1, 0, 4, {}},

                    Face{3, 6, 2, {}}, // right
                    Face{3, 7, 6, {}},
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
    RunLoop() : globalConfig(rendererType, windowSize, fov), sdlController(SDLController(globalConfig)) {
        disposePool.push_back(windowSize.addObserver(
            [](std::pair<size_t, size_t> screenSize) {
                std::cout << "New screen size " << screenSize.first << " x " << screenSize.second << std::endl;
            }
        ));
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
                    windowSize.value({event.window.data1, event.window.data2});
                }
                break;
            default:
                break;
            }
        }
    }

    ObservableProperty<RendererType> rendererType{RendererType::CPU};
    ObservableProperty<std::pair<size_t, size_t>> windowSize{{800, 600}};
    ObservableProperty<float> fov{1.3962634016};

    GlobalConfig globalConfig;
    SDLController sdlController;

    bool shouldClose = false;
    unsigned long long previousFrameTicks = 0;
    const int frameRate = 60;
    const int frameTime = 1000 / frameRate;

    DisposePool disposePool;
};