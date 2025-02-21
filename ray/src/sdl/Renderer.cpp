#include <cmath>
#include <iostream>
#include <limits>
#include <memory>

#include "MathUtils.hpp"
#include "Utility.hpp"
#include "sdl/Camera.hpp"
#include "sdl/Renderer.hpp"

template <typename T>
static void safeMemorySet(
    T *p, T const &value, size_t n
) {
    for (size_t i = 0; i < n; i++) {
        p[i] = value;
    }
}

sdl::Renderer::Renderer(
    SDL_Window *window,
    std::pair<size_t, size_t> resolution,
    Lazy<sdl::Camera> camera
)
    : renderer(SDL_CreateRenderer(window, -1, 0)),
      camera(camera),
      resolution(resolution) {

    light = {sdl::light::DirectionalLight{{0, -0.7071067812, -0.7071067812}}};

    auto resolutionSize = resolution.first * resolution.second;
    colorBuffer = std::unique_ptr<uint32_t[]>(new uint32_t[resolutionSize]);
    zBuffer = std::unique_ptr<float[]>(new float[resolutionSize]);
    safeMemorySet(zBuffer.get(), std::numeric_limits<float>::max(), resolutionSize);

    screenSpaceProjection_ = ml::screenSpaceProjection(resolution.first, resolution.second);
    renderTarget = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_RGBA8888,
        SDL_TEXTUREACCESS_STREAMING,
        resolution.first,
        resolution.second
    );
}

sdl::Renderer::~Renderer() {
    SDL_DestroyRenderer(renderer);
}

void sdl::Renderer::processInput(void const *eventData) {
    SDL_Event event = *reinterpret_cast<SDL_Event const *>(eventData);
    static float const cameraSpeed = 0.5f;
    switch (event.type) {
    case SDL_KEYDOWN:
        switch (event.key.keysym.sym) {
            case SDLK_w:
            case SDLK_a:
            case SDLK_d:
            case SDLK_s:
                camera()->handleInput(
                    CameraInput::Translate::make(event.key.keysym.sym, cameraSpeed)
                );
        }
    // case SDL
    default:
        break;
    }
}

void sdl::Renderer::update(
    MeshData const &data, float dt
) {
    ml::Matrix4f const perspectiveProjectionMatrix = camera()->getScenePerspectiveProjectionMatrix();
    ml::Matrix4f const cameraTransformation = camera()->getViewTransformation();
    for (auto const &node : data) {
        auto const &mesh = node.meshBuffer;
        auto const transformMatrix = node.getTransform();
        for (size_t i = 0; i < mesh.faces.size(); i++) {
            auto const &face = mesh.faces[i];

            auto vertexA = ml::matMul(transformMatrix, ml::as<4, 3, float>(mesh.vertices[face.a], 1.f));
            auto vertexB = ml::matMul(transformMatrix, ml::as<4, 3, float>(mesh.vertices[face.b], 1.f));
            auto vertexC = ml::matMul(transformMatrix, ml::as<4, 3, float>(mesh.vertices[face.c], 1.f));

            vertexA = ml::matMul(cameraTransformation, vertexA);
            vertexB = ml::matMul(cameraTransformation, vertexB);
            vertexC = ml::matMul(cameraTransformation, vertexC);

            ml::Vector3f faceNormal = ml::normalized(ml::crossProduct(
                ml::as<3, 4, float>(vertexB) - ml::as<3, 4, float>(vertexA),
                ml::as<3, 4, float>(vertexC) - ml::as<3, 4, float>(vertexA)
            ));

            if (ml::dotProduct(faceNormal, -ml::as<3, 4, float>(vertexA)) <= 0.f) {
                continue;
            }

            // perspective projection
            ml::Vector4f projectedPoints[] = {
                ml::matMul(perspectiveProjectionMatrix, vertexA),
                ml::matMul(perspectiveProjectionMatrix, vertexB),
                ml::matMul(perspectiveProjectionMatrix, vertexC)
            };

            float z0 = projectedPoints[0][3], z1 = projectedPoints[1][3], z2 = projectedPoints[2][3];
            for (size_t i = 0; i < 3; i++) {
                projectedPoints[i] = ml::matrixScale(projectedPoints[i], 1 / projectedPoints[i][3]);
            }

            // screen space projection
            ml::Vector4f screenProjectedPoints[] = {
                ml::matMul(screenSpaceProjection_, projectedPoints[0]),
                ml::matMul(screenSpaceProjection_, projectedPoints[1]),
                ml::matMul(screenSpaceProjection_, projectedPoints[2])
            };

            Triangle tri = {
                {screenProjectedPoints[0],
                 screenProjectedPoints[1],
                 screenProjectedPoints[2]
                },
                face.attributes
            };

            auto const lightInverseDirection = ml::matrixScale(std::get<sdl::light::DirectionalLight>(light).direction, -1.f);
            auto const lightIntensity = ml::cosineSimilarity(faceNormal, lightInverseDirection);
            uint32_t color = interpolateRGBAColorIntensity(
                0x00FF0088,
                lightIntensity,
                0.1f
            );

            auto const renderMethod = RenderMethod::fill;
            switch (renderMethod) {
            case RenderMethod::vertices:
                for (size_t j = 0; j < 3; j++) {
                    drawPoint(color, {tri.vertices[j][0], tri.vertices[j][1]}, 3);
                }
                break;
            case RenderMethod::wireframe:
                drawLine({tri.vertices[0][0], tri.vertices[0][1]}, {tri.vertices[1][0], tri.vertices[1][1]}, color);
                drawLine({tri.vertices[1][0], tri.vertices[1][1]}, {tri.vertices[2][0], tri.vertices[2][1]}, color);
                drawLine({tri.vertices[2][0], tri.vertices[2][1]}, {tri.vertices[0][0], tri.vertices[0][1]}, color);
                for (size_t j = 0; j < 3; j++) {
                    drawPoint(color, {tri.vertices[j][0], tri.vertices[j][1]}, 3);
                }
                break;
            case RenderMethod::fill:
                fillTriangle(tri, z0, z1, z2, color);
                break;
            }
        }
    }
}

void sdl::Renderer::render() const {
    auto const &[w, h] = resolution;
    SDL_UpdateTexture(
        renderTarget,
        nullptr,
        colorBuffer.get(),
        w * sizeof(uint32_t)
    );

    SDL_RenderCopy(renderer, renderTarget, nullptr, nullptr);
    safeMemorySet(colorBuffer.get(), (uint32_t)0x000000FF, w * h);
    safeMemorySet(zBuffer.get(), std::numeric_limits<float>::max(), w * h);
    SDL_RenderPresent(renderer);
}

void sdl::Renderer::drawPoint(
    uint32_t color, ml::Vector2i position, size_t thickness
) noexcept {
    if (thickness == 0) {
        colorBuffer[position.x + position.y * resolution.first] = color;
        return;
    }

    int thick = thickness;
    for (int i = -thick; i < thick; i++) {
        for (int j = -thick; j < thick; j++) {
            colorBuffer[position.x + i + (position.y + j) * resolution.first] = color;
        }
    }
}

static constexpr float fpart(
    float num
) {
    return num - static_cast<int>(num);
}

static constexpr float oneComplement(
    float num
) {
    return 1 - fpart(num);
}

void sdl::Renderer::drawLine(
    ml::Vector2i from, ml::Vector2i to, uint32_t color
) noexcept {
    int x0{from.x},
        y0{from.y},
        x1{to.x},
        y1{to.y};
    bool isSteep = std::abs(from.y - to.y) > std::abs(from.x - to.x);
    if (isSteep) {
        std::swap(x0, y0);
        std::swap(x1, y1);
    }
    if (x0 > x1) {
        std::swap(x0, x1);
        std::swap(y0, y1);
    }
    int dx{x1 - x0}, dy{y1 - y0};
    float slope = dx == 0
                      ? dy / std::abs(dy)
                      : static_cast<float>(dy) / dx;

    std::function<void(size_t x, size_t y, uint32_t value)> assign = [p = colorBuffer.get(), isSteep, res = resolution](size_t x, size_t y, uint32_t value) {
        if (isSteep) {
            std::swap(x, y);
        }
        p[x + y * res.first] = value;
    };

    int x = x0;
    float y = y0;
    while (x <= x1) {
        int y_ = y;
        assign(x, y_, interpolateRGBAColorIntensity(color, oneComplement(std::abs(y))));
        assign(x, y_ + 1, interpolateRGBAColorIntensity(color, fpart(std::abs(y))));
        x++;
        y += slope;
    }
}

void sdl::Renderer::fillTriangle(
    Triangle tri,
    float z0,
    float z1,
    float z2,
    uint32_t color
) noexcept {
    auto &t0 = tri.vertices[0];
    auto &t1 = tri.vertices[1];
    auto &t2 = tri.vertices[2];
    if (t0.y > t1.y) {
        std::swap(t0, t1);
        std::swap(z0, z1);
    }
    if (t1.y > t2.y) {
        std::swap(t1, t2);
        std::swap(z1, z2);
    }
    if (t0.y > t1.y) {
        std::swap(t0, t1);
        std::swap(z0, z1);
    }

    int x0 = t0.x, y0 = t0.y;
    int x1 = t1.x, y1 = t1.y;
    int x2 = t2.x, y2 = t2.y;

    // According to rasterization rules:
    // https://learn.microsoft.com/en-us/windows/win32/direct3d11/d3d10-graphics-programming-guide-rasterizer-stage-rules
    // Do not draw bottom and right edge
    if (y0 != y1) {
        float inv_slope0 = static_cast<float>(x1 - x0) / (y1 - y0);
        float inv_slope1 = static_cast<float>(x2 - x0) / (y2 - y0);
        for (int y = y0; y < y1; y++) {
            int xBegin = x0 + (y - y0) * inv_slope0;
            int xEnd = x0 + (y - y0) * inv_slope1;
            if (xBegin > xEnd) {
                std::swap(xBegin, xEnd);
            }
            for (int x = xBegin; x < xEnd; x++) {
                auto weights = ml::barycentricWeights(
                    {x0, y0},
                    {x1, y1},
                    {x2, y2},
                    {x, y}
                );
                auto zValue = ml::perspectiveInterpolate(
                    z0, z1, z2, weights
                );
                auto texelIdx = x + y * resolution.first;
                if (zBuffer[texelIdx] > zValue) {
                    zBuffer[texelIdx] = zValue;
                    colorBuffer[texelIdx] = color;
                }
            }
        }
    }
    if (y1 != y2) {
        float inv_slope0 = static_cast<float>(x0 - x2) / (y2 - y0);
        float inv_slope1 = static_cast<float>(x1 - x2) / (y2 - y1);
        for (int y = y2; y >= y1; y--) {
            int xBegin = x2 + (y2 - y) * inv_slope0;
            int xEnd = x2 + (y2 - y) * inv_slope1;
            if (xBegin > xEnd) {
                std::swap(xBegin, xEnd);
            }
            for (int x = xBegin; x < xEnd; x++) {
                auto weights = ml::barycentricWeights(
                    {x0, y0},
                    {x1, y1},
                    {x2, y2},
                    {x, y}
                );
                auto zValue = ml::perspectiveInterpolate(
                    z0, z1, z2, weights
                );
                auto texelIdx = x + y * resolution.first;
                if (zBuffer[texelIdx] > zValue) {
                    zBuffer[texelIdx] = zValue;
                    colorBuffer[texelIdx] = color;
                }
            }
        }
    }
}