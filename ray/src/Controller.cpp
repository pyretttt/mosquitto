#include <memory>

#include "SDL.h"

#include "Core.hpp"
#include "Errors.hpp"
#include "ReactivePrimitives.hpp"
#include "Controller.hpp"
#include "Utility.hpp"
#include "sdl/Renderer.hpp"
#include "Lazy.hpp"
#include "sdl/Camera.hpp"

Controller::Controller(
    std::shared_ptr<GlobalConfig> config
) : config(config) {
    connections.push_back(
        std::move(config->rendererType.asObservableObject()->subscribe([this](auto rendererType) {
            this->prepare(this->config);
        }))
    );
    prepare(config);
}

Controller::~Controller() {}

void Controller::prepare(std::shared_ptr<GlobalConfig> config) {
    switch (config->rendererType.value()) {
    case (RendererType::CPU):
        renderer = std::make_shared<sdl::Renderer>(
            config->windowSize.value(),
            Lazy<sdl::Camera>([config = this->config]() {
                return std::make_shared<sdl::Camera>(config->fov.asObservableObject(), config->windowSize.asObservableObject());
            })
        );
        break;
    case (RendererType::OpenGL):
        renderer = nullptr;    
        break;
    }
    renderer->prepareViewPort();
}
