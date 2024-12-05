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
    void drawPoint(uint32_t color, Vector2i position, size_t thickness = 0) noexcept;
    void drawLine(Vector2i from, Vector2i to, uint32_t color) noexcept;
    void fillTriangle(Triangle t) noexcept;

    std::pair<int, int> resolution;
    SDL_Renderer *renderer;
    std::unique_ptr<uint32_t []> colorBuffer;
    std::unique_ptr<uint32_t []> zBuffer;
    SDL_Texture *renderTarget;
    Matrix4f perspectiveProjectionMatrix_;
    Matrix4f screenSpaceProjection_;
};