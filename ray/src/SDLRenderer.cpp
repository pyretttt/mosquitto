#include <iostream>
#include <memory>

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

void SDLRenderer::update(MeshData const &data, float dt) {
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
                drawPoint(0xFFF11FFF, {vertex.x(), vertex.y()}, 2);
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
    memset(colorBuffer.get(), (uint32_t)0xFF000000, w * h * sizeof(uint32_t));
    SDL_RenderPresent(renderer);
}

void SDLRenderer::drawPoint(uint32_t color, Eigen::Vector2i position, size_t thickness) noexcept {
    int thick = thickness;
    for (int i = -thick; i < thick; i++) {
        for (int j = -thick; j < thick; j++) {
            colorBuffer[position.x() + i + (position.y() + j) * resolution.first] = color;
        }
    }
}

void SDLRenderer::drawLine(Eigen::Vector2i from, Eigen::Vector2i to, uint32_t color) noexcept {
    int x0{from.x()},
        y0{from.y()},
        x1{to.x()},
        y1{to.y()};
    bool isSteep = std::abs(from.y() - to.y()) > std::abs(from.x() - to.x());
    if (isSteep) {
        std::swap(x0, y0);
        std::swap(x1, y1);
    }
    if (x0 > x1) {
        std::swap(x0, x1);
        std::swap(y0, y1);
    }
    int dx{x1 - x0}, dy{y1 - y0};
    float slope = dx == 0 ? 1.f : dy / dx;
    float interY = y0;
    
    std::function<void(size_t x, size_t y, uint32_t value)> assign = [p = colorBuffer.get(), isSteep, res = resolution](size_t x, size_t y, uint32_t value) {
        if (isSteep) {
            std::swap(x, y);
        }
        p[x + y * res.first] = value;
    };

    
}

static constexpr float fpart(float num) {
    return num - static_cast<int>(num);
}

static constexpr float oneComplement(float num) {
    return 1 - fpart(num);
}