#pragma once

#include <functional>
#include <utility>

#include "SDL.h"

#include "Renderer.h"
#include "Mesh.h"
#include "MathUtils.h"

struct SDLRenderer : public Renderer {
    SDLRenderer(SDL_Window *window, std::pair<size_t, size_t> resolution);
    void update(MeshData const &data, float dt) override;
    void render() const override;

    ~SDLRenderer();
private:
    void drawPoint(uint32_t color, ml::Vector2i position, size_t thickness = 0) noexcept;
    void drawLine(ml::Vector2i from, ml::Vector2i to, uint32_t color) noexcept;
    void fillTriangle(Triangle t) noexcept;

    std::pair<int, int> resolution;
    SDL_Renderer *renderer;
    std::unique_ptr<uint32_t []> colorBuffer;
    std::unique_ptr<uint32_t []> zBuffer;
    SDL_Texture *renderTarget;
    ml::Matrix4f perspectiveProjectionMatrix_;
    ml::Matrix4f screenSpaceProjection_;
};