#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <array>

#include "Core.hpp"
#include "MathUtils.hpp"
#include "Utility.hpp"
#include "Camera.hpp"
#include "sdl/Renderer.hpp"
#include "Polygon.hpp"

template <typename T>
static void safeMemorySet(
    T *p, T const &value, size_t n
) {
    for (size_t i = 0; i < n; i++) {
        p[i] = value;
    }
}

sdl::Renderer::Renderer(
    std::shared_ptr<GlobalConfig> config,
    Lazy<Camera> camera
) 
    : camera(camera)
    , config(config)
    , resolution(config->windowSize.value()) {

    light = {sdl::light::DirectionalLight{{0, -0.7071067812, -0.7071067812}}};

    auto resolutionSize = resolution.first * resolution.second;
    colorBuffer = std::unique_ptr<uint32_t[]>(new uint32_t[resolutionSize]);
    zBuffer = std::unique_ptr<float[]>(new float[resolutionSize]);
    safeMemorySet(zBuffer.get(), std::numeric_limits<float>::max(), resolutionSize);

    screenSpaceProjection_ = ml::screenSpaceProjection(resolution.first, resolution.second);
}

void sdl::Renderer::prepareViewPort() {
    auto window = SDL_CreateWindow(
        nullptr,
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        resolution.first,
        resolution.second,
        SDL_WINDOW_BORDERLESS | SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL
    );
    config->window.reset(window);
    renderer = SDL_CreateRenderer(window, -1, 0);
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

void sdl::Renderer::processInput(Event event) {
    static float const cameraSpeed = 0.25f;
    std::visit(overload {
        [&camera = this->camera, resolution = this->resolution](SDL_Event event) {
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
                        break;
                    case SDLK_f:
                        auto const currentValue = SDL_ShowCursor(SDL_QUERY);
                        SDL_ShowCursor(currentValue == SDL_ENABLE ? SDL_DISABLE : SDL_ENABLE);
                        break;
                }
            case SDL_MOUSEMOTION:
                auto const isTrackingMoution = SDL_ShowCursor(SDL_QUERY) == SDL_DISABLE;
                if (isTrackingMoution) {
                    // Some strange behavior when cursor is switched it reports huge delta
                    if (std::abs(event.motion.xrel) > (int32_t)resolution.second
                    || std::abs(event.motion.yrel) > (int32_t)resolution.first) {
                        break;
                    }
                    camera()->handleInput(
                        CameraInput::Rotate {
                            .delta = std::make_pair(
                                event.motion.yrel,
                                event.motion.xrel
                            )
                        }
                    );
                }
                break;
            }
        }
    }, event);
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

            std::array<Attributes::Cases, 3> attributes = {
                mesh.attributes[face.a],
                mesh.attributes[face.b],
                mesh.attributes[face.c],
            };
            auto polygon = Polygon::fromTriangle(
                Triangle(
                    std::to_array<ml::Vector4f>(projectedPoints),
                    attributes
                )
            );
            polygon.clip();
            auto const triangles = polygon.getTriangles();

            for (size_t triangleIdx = 0; triangleIdx < polygon.numTriangles(); triangleIdx++) {
                auto triangleToRender = triangles[triangleIdx];
                float z0 = triangleToRender.vertices[0][3], z1 = triangleToRender.vertices[1][3], z2 = triangleToRender.vertices[2][3];
                for (size_t i = 0; i < 3; i++) {
                    triangleToRender.vertices[i] = ml::matrixScale(triangleToRender.vertices[i], 1 / triangleToRender.vertices[i][3]);
                }
                
                for (size_t i = 0; i < 3; i++) {
                    triangleToRender.vertices[i] = ml::matMul(screenSpaceProjection_, triangleToRender.vertices[i]);
                }

                auto const lightInverseDirection = ml::matrixScale(std::get<sdl::light::DirectionalLight>(light).direction, -1.f);
                auto const lightIntensity = ml::cosineSimilarity(faceNormal, lightInverseDirection);
                uint32_t color = interpolateRGBAColorIntensity(
                    0x00FF0088,
                    lightIntensity,
                    0.1f
                );

                auto const renderMethod = RenderMethod::wireframe;
                switch (renderMethod) {
                case RenderMethod::vertices:
                    for (size_t j = 0; j < 3; j++) {
                        drawPoint(color, {triangleToRender.vertices[j][0], triangleToRender.vertices[j][1]}, 3);
                    }
                    break;
                case RenderMethod::wireframe:
                    drawLine({triangleToRender.vertices[0][0], triangleToRender.vertices[0][1]}, {triangleToRender.vertices[1][0], triangleToRender.vertices[1][1]}, color);
                    drawLine({triangleToRender.vertices[1][0], triangleToRender.vertices[1][1]}, {triangleToRender.vertices[2][0], triangleToRender.vertices[2][1]}, color);
                    drawLine({triangleToRender.vertices[2][0], triangleToRender.vertices[2][1]}, {triangleToRender.vertices[0][0], triangleToRender.vertices[0][1]}, color);
                    for (size_t j = 0; j < 3; j++) {
                        drawPoint(color, {triangleToRender.vertices[j][0], triangleToRender.vertices[j][1]}, 3);
                    }
                    break;
                case RenderMethod::fill:
                    fillTriangle(triangleToRender, z0, z1, z2, color);
                    break;
                }
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
            size_t y = position.y + j;
            size_t x = position.x + i;
            if (x >= resolution.first || y >= resolution.second) { continue; }
            colorBuffer[x + y * resolution.first] = color;
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
        if (x >= res.first || y >= res.second) { return; }
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