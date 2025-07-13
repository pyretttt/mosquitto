#include <iostream>
#include <any>
#include <unordered_map>

#include "SDL.h"
#include <yaml-cpp/yaml.h>

#include "GlobalConfig.hpp"
#include "MathUtils.hpp"
#include "ReactivePrimitives.hpp"
#include "RendererBase.hpp"
#include "Controller.hpp"

namespace {
    YAML::Node config = YAML::LoadFile("Config.yml");

    GlobalConfig makeGlobalConfig(YAML::Node const &config) {
        static const std::unordered_map<std::string, RendererType> rendererMap = {
            {"gl", RendererType::OpenGL},
            {"cpu", RendererType::CPU}
        };
        if (config.IsDefined()) {
            auto renderer = rendererMap.at(config["renderer"].as<std::string>());
            auto window = std::make_pair<size_t, size_t>(
                config["window"]["width"].as<size_t>(),
                config["window"]["height"].as<size_t>()
            );
            auto degrees = config["fov"].as<size_t>();
            auto cameraSpeed = config["camera_speed"].as<float>();
            auto mouseSpeed = config["mouse_speed"].as<float>();

            return GlobalConfig(
                ObservableProperty<RendererType>(RendererType::OpenGL),
                ObservableProperty<std::pair<size_t, size_t>>({800, 600}),
                ObservableProperty<float>(ml::toRadians(80)),
                ObservableProperty<float>(cameraSpeed),
                ObservableProperty<float>(mouseSpeed)
            );
        }
        throw std::invalid_argument("Invalid congif");
    }
}

class RunLoop {
public:
    RunLoop(RunLoop const &other) = delete;
    RunLoop &operator=(RunLoop const &other) = delete;

    inline static RunLoop &instance() {
        static RunLoop loop;
        return loop;
    }

    void start() {
        

        Renderer::MeshData node = Renderer::MeshData{
            MeshNode(
                MeshBuffer(
                    std::vector<ml::Vector3f>{
                        ml::Vector3f(-0.5, 0.5, 0.5),   // 0. left - top - near
                        ml::Vector3f(-0.5, -0.5, 0.5),  // 1. left - bottom - near
                        ml::Vector3f(0.5, 0.5, 0.5),    // 2. right - top - near
                        ml::Vector3f(0.5, -0.5, 0.5),   // 3. right - bottom - near
                        ml::Vector3f(-0.5, 0.5, -0.5),  // 4. left - top - far
                        ml::Vector3f(-0.5, -0.5, -0.5), // 5. left - bottom - far
                        ml::Vector3f(0.5, 0.5, -0.5),   // 6. right - top - far
                        ml::Vector3f(0.5, -0.5, -0.5),  // 7. right - bottom - far
                    },
                    std::vector<Face>{
                        Face{5, 6, 7}, // far front
                        Face{5, 4, 6},

                        Face{1, 2, 0}, // near front
                        Face{1, 3, 2},

                        Face{0, 6, 4}, // top
                        Face{0, 2, 6},

                        Face{1, 7, 3}, // bottom
                        Face{1, 5, 7},

                        Face{1, 4, 5}, // left
                        Face{1, 0, 4},

                        Face{3, 6, 2}, // right
                        Face{3, 7, 6},
                    },
                    std::vector<Attributes::Cases>(8, Attributes::Cases())
                )
            )
        };

        while (!shouldClose) {
            auto currentTicks = SDL_GetTicks();
            auto dt = currentTicks - previousFrameTicks;
            if (dt < frameTime) {
                SDL_Delay(frameTime - dt);
                dt = SDL_GetTicks() - previousFrameTicks;
            }
            previousFrameTicks = currentTicks;
            auto dtFraction = dt / 1000.f;
            // std::cout << "Dt " << dt << std::endl;
            ml::Matrix4f transformationMatrix = ml::scaleMatrix(15, 15, 15);
            transformationMatrix = ml::matMul(
                ml::rodriguezRotationMatrix({0, 1, 0}, static_cast<float>(freezeTick.has_value() ? freezeTick.value() : currentTicks) / 1000),
                transformationMatrix
            );
            transformationMatrix = ml::matMul(ml::translationMatrix(0, -20, -50), transformationMatrix);
            node[0].transform = transformationMatrix;
            
            processInput(dtFraction);
            controller.renderer->update(node, dtFraction);
            controller.renderer->render();
        }
    }

private:
    RunLoop() {}

    inline void processInput(float dt) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
            case SDL_PRESSED:
                switch (event.key.keysym.sym) {
                    case SDLK_w:
                        break;
                }
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
                case SDLK_SPACE:
                    if (freezeTick.has_value()) {
                        freezeTick.reset();
                    } else {
                        freezeTick = SDL_GetTicks();
                    }
                }
                break;
            case SDL_QUIT:
            case SDL_WINDOWEVENT_CLOSE:
                shouldClose = true;
                break;
            case SDL_WINDOWEVENT:
                switch (event.window.event) {
                case SDL_WINDOWEVENT_RESIZED:
                    globalConfig->windowSize.value({event.window.data1, event.window.data2});
                    break;
                }
                break;
            default:
                break;
            }

            controller.renderer->processInput(Event(event), dt);
        }
    }

    std::shared_ptr<GlobalConfig> globalConfig = std::make_shared<GlobalConfig>(
        makeGlobalConfig(config)
    );
    Controller controller = Controller(globalConfig);

    bool shouldClose = false;
    unsigned long long previousFrameTicks = 0;
    const int frameRate = 120;
    const int frameTime = 1000 / frameRate;
    std::optional<unsigned long long> freezeTick;
};