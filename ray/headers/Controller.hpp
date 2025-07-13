#pragma once

#include <functional>
#include <utility>

#include "SDL.h"

#include "GlobalConfig.hpp"
#include "RendererBase.hpp"

struct Controller {
    Controller() = delete;
    Controller(Controller const &other) = delete;
    Controller(Controller &&other) = delete;
    Controller operator=(Controller const &other) = delete;
    Controller operator=(Controller &&other) = delete;
    explicit Controller(
        std::shared_ptr<GlobalConfig> config
    );
    ~Controller();
    
    
    std::shared_ptr<Renderer> renderer;
private:
    void prepare(std::shared_ptr<GlobalConfig> config);

    std::shared_ptr<GlobalConfig> config;
    std::vector<std::unique_ptr<Connection>> connections;
};