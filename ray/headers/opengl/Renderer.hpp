#pragma once

#include <utility>
#include <any>

#include "RendererBase.hpp"

#include "Lazy.hpp"
#include "Camera.hpp"

namespace gl {
struct Renderer: public ::Renderer {
    Renderer() = delete;
    Renderer(Renderer &&other) = delete;
    Renderer(Renderer const &other) = delete;
    Renderer operator=(Renderer &&other) = delete;
    Renderer operator=(Renderer const &other) = delete;
    Renderer(std::shared_ptr<GlobalConfig>, Lazy<Camera> camera);
    void prepareViewPort() override;
    void processInput(Event, float dt) override;
    void update(MeshData const &data, float dt) override;
    void render() const override;

    ~Renderer() override;

    std::pair<size_t, size_t> resolution;
    std::shared_ptr<GlobalConfig> config;
    Lazy<Camera> camera;
    SDL_GLContext glContext;
};
}