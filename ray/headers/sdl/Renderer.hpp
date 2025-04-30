#pragma once

#include <functional>
#include <utility>

#include "SDL.h"

#include "Lazy.hpp"
#include "MathUtils.hpp"
#include "Mesh.hpp"
#include "RendererBase.hpp"
#include "sdl/Light.hpp"

namespace sdl {

class Camera;

struct Renderer : public ::Renderer {
    Renderer() = delete;
    Renderer(Renderer &&other) = delete;
    Renderer(Renderer const &other) = delete;
    Renderer operator=(Renderer &&other) = delete;
    Renderer operator=(Renderer const &other) = delete;
    Renderer(std::pair<size_t, size_t> resolution, Lazy<Camera> camera);
    void prepareViewPort() override;
    void processInput(void const *) override;
    void update(MeshData const &data, float dt) override;
    void render() const override;

    ~Renderer() override;

private:
    void drawPoint(uint32_t color, ml::Vector2i position, size_t thickness = 0) noexcept;
    void drawLine(ml::Vector2i from, ml::Vector2i to, uint32_t color) noexcept;
    void fillTriangle(Triangle t, float z0, float z1, float z2, uint32_t color) noexcept;

    std::pair<size_t, size_t> resolution;
    Lazy<sdl::Camera> camera;
    std::unique_ptr<uint32_t[]> colorBuffer;
    std::unique_ptr<float[]> zBuffer;
    SDL_Texture *renderTarget;
    ml::Matrix4f screenSpaceProjection_;
    light::Cases light;

    SDL_Window *window = nullptr;
    SDL_Renderer *renderer = nullptr;
};
} // namespace sdl