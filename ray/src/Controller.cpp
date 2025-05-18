#include <memory>

#include "SDL.h"

#include "Core.hpp"
#include "Errors.hpp"
#include "ReactivePrimitives.hpp"
#include "Controller.hpp"
#include "Utility.hpp"
#include "sdl/Renderer.hpp"
#include "opengl/Renderer.hpp"
#include "Lazy.hpp"
#include "Camera.hpp"
#include "GlobalConfig.hpp"

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
    auto camera = Lazy<Camera>([config = this->config]() {
        return std::make_shared<Camera>(
            config->fov.asObservableObject(),
            config->windowSize.asObservableObject(),
            config->rendererType.asObservableObject()
        );
    });
    switch (config->rendererType.value()) {
    case (RendererType::CPU):
        renderer = std::make_shared<sdl::Renderer>(
            config,
            camera
        );
        break;
    case (RendererType::OpenGL):
        renderer = std::make_shared<gl::Renderer>(
            config,
            camera
        );
        break;
    }
    renderer->prepareViewPort();
}
