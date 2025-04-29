#include <memory>

#include "SDL.h"

#include "Core.hpp"
#include "Errors.hpp"
#include "ReactivePrimitives.hpp"
#include "RendererFactory.hpp"
#include "Controller.hpp"
#include "Utility.hpp"
#include "sdl/Renderer.hpp"

Controller::Controller(
    std::shared_ptr<GlobalConfig> config
) : config(config) {
    renderer = makeRenderer(RenderFactoryParams(config));
    // config.rendererType.asObservableObject()->subscribe([config](auto rendererType) {
    //     makeRenderer(RenderFactoryParams(config));
    // });
}

Controller::~Controller() {}

void Controller::prepare() {
    renderer = makeRenderer(RenderFactoryParams(config));
    renderer->prepareViewPort();
}
