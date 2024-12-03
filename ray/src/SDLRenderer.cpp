#include "memory"
#include <iostream>

#include "MathUtils.h"
#include "SDLRenderer.h"

SDLRenderer::SDLRenderer(SDL_Window *window, std::pair<int, int> resolution)
    : renderer(SDL_CreateRenderer(window, -1, 0)),
      resolution(resolution) {

    auto resolutionSize = resolution.first * resolution.second;
    colorBuffer = std::unique_ptr<uint32_t[]>(new uint32_t[resolutionSize]);
    zBuffer = std::unique_ptr<uint32_t[]>(new uint32_t[resolutionSize]);
    renderTarget = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STREAMING,
        resolution.first,
        resolution.second
    );
}

SDLRenderer::~SDLRenderer() {
    SDL_DestroyRenderer(renderer);
}

void SDLRenderer::update(MeshData const &data) {
    for (auto const &mesh : data) {
        for (size_t i = 0; i < mesh.faces.size(); i++) {
            auto const &face = mesh.faces[i];
            Triangle tri = Triangle{
                {asVec4(mesh.vertices[face.a], 0),
                 asVec4(mesh.vertices[face.b], 0),
                 asVec4(mesh.vertices[face.c], 0)},
                face.uv // std::move() ?
            };

            for (auto const vertex : tri.vertices) {
                drawPoint(0xFFFFFFFF, {vertex.x(), vertex.y()}, 2);
            }
        }
    }
}

void SDLRenderer::render() const {
    auto const &[w, h] = resolution;
    SDL_UpdateTexture(
        renderTarget,
        nullptr,
        colorBuffer.get(),
        w * sizeof(uint32_t)
    );
    SDL_RenderCopy(renderer, renderTarget, nullptr, nullptr);
    memset(colorBuffer.get(), 0xFF000000, w * h);
}

void SDLRenderer::drawPoint(uint32_t color, Eigen::Vector2i position, size_t thickness) noexcept {
    for (int i = -thickness; i < thickness; i++) {
        for (int j = -thickness; j < thickness; j++) {
            colorBuffer[position.x() + i
                + (position.y() + j) * resolution.first] = color;
        }
    }
}