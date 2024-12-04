#pragma once

#include <functional>
#include <utility>

#include "SDL.h"

#include "Renderer.h"
#include "Mesh.h"

struct SDLRenderer : public Renderer {
    SDLRenderer(SDL_Window *window, std::pair<int, int> resolution);
    void update(MeshData const &data, float dt) override;
    void render() const override;

    ~SDLRenderer();

private:
    void drawPoint(uint32_t color, Eigen::Vector2i position, size_t thickness = 0) noexcept;
    void drawLine(Eigen::Vector2i from, Eigen::Vector2i to, uint32_t color) noexcept;
    void fillTriangle(Triangle t) noexcept;

    std::pair<int, int> resolution;
    SDL_Renderer *renderer;
    std::unique_ptr<uint32_t []> colorBuffer;
    std::unique_ptr<uint32_t []> zBuffer;
    SDL_Texture *renderTarget;
};