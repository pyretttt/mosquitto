#pragma once

#include <functional>
#include <utility>

#include "SDL.h"

#include "MathUtils.h"
#include "Mesh.h"
#include "RendererBase.h"
#include "sdl/Light.h"
#include "Lazy.h"

namespace sdl {

class Camera;

struct Renderer : public ::Renderer {
    Renderer(SDL_Window *window, std::pair<size_t, size_t> resolution, Lazy<Camera> camera);
    void update(MeshData const &data, float dt) override;
    void render() const override;

    ~Renderer();

private:
    void drawPoint(uint32_t color, ml::Vector2i position, size_t thickness = 0) noexcept;
    void drawLine(ml::Vector2i from, ml::Vector2i to, uint32_t color) noexcept;
    void fillTriangle(Triangle t, uint32_t color) noexcept;

    std::pair<size_t, size_t> resolution;
    Lazy<Camera> camera;
    SDL_Renderer *renderer;
    std::unique_ptr<uint32_t[]> colorBuffer;
    std::unique_ptr<uint32_t[]> zBuffer;
    SDL_Texture *renderTarget;
    ml::Matrix4f perspectiveProjectionMatrix_;
    ml::Matrix4f screenSpaceProjection_;
    light::Cases light;
};
} // namespace sdl