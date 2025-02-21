#pragma once

#include <utility>
#include <variant>

#include "SDL.h"

#include "MathUtils.hpp"
#include "ReactivePrimitives.hpp"

namespace sdl {

struct CameraInput {
    struct Translate {
        float forward, backward, left, right;

        static inline Translate make(SDL_Keycode keyCode, float speed) {
            Translate t{
                .forward = 0,
                .backward = 0,
                .left = 0,
                .right = 0
            };
            switch (keyCode) {
                case SDLK_w:
                    t.forward = speed;
                    return t;
                case SDLK_a:
                    t.left = speed;
                    return t;
                case SDLK_d:
                    t.right = speed;
                    return t;
                case SDLK_s:
                    t.backward = speed;
                    return t;
                default:
                    return t;
            }
        }
    };
    struct Rotate {
        std::array<ml::Vector2f, 3> uv;
    };

    using Cases = std::variant<Translate, Rotate>;
};

class Camera final {
public:
    Camera() = delete;
    Camera(Camera &&other) = delete;
    Camera(Camera const &other) = delete;
    Camera operator=(Camera &&other) = delete;
    Camera operator=(Camera const &other) = delete;

    explicit Camera(
        std::unique_ptr<ObservableObject<float>> &&fov,
        std::unique_ptr<ObservableObject<std::pair<size_t, size_t>>> &&windowSize
    );
    ml::Matrix4f const &getScenePerspectiveProjectionMatrix() const noexcept;
    ml::Matrix4f const &getViewTransformation() const noexcept;

    void handleInput(CameraInput::Cases const &cameraInput) noexcept;

private:
    ml::Vector3f origin;
    ml::Vector3f lookAt = {0, 0, -1};
    // ml::Vector3f up;
    // ml::Vector3f right;
    std::unique_ptr<ObservableObject<float>> fov_;
    std::unique_ptr<ObservableObject<std::pair<size_t, size_t>>> windowSize_;
    ml::Matrix4f viewMatrix;
    ml::Matrix4f perspectiveProjectionMatrix;
    Connections connections;
};
} // namespace sdl