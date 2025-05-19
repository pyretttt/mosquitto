#pragma once

#include <utility>
#include <variant>
#include <vector>

#include "SDL.h"

#include "MathUtils.hpp"
#include "ReactivePrimitives.hpp"

enum class RendererType;

struct CameraInput {
    struct Translate {
        bool forward, backward, left, right;

        static inline Translate make(SDL_Keycode keyCode) {
            Translate t{
                .forward = false,
                .backward = false,
                .left = false,
                .right = false
            };
            switch (keyCode) {
                case SDLK_w:
                    t.forward = true;
                    break;
                case SDLK_a:
                    t.left = true;
                    break;
                case SDLK_d:
                    t.right = true;
                    break;
                case SDLK_s:
                    t.backward = true;
                    break;
                default:
                    break;
            }
            return t;
        }
    };
    struct Rotate {
        std::pair<int32_t, int32_t> delta;
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
        std::unique_ptr<ObservableObject<float>> fov,
        std::unique_ptr<ObservableObject<std::pair<size_t, size_t>>> windowSize,
        std::unique_ptr<ObservableObject<RendererType>> rendererType,
        std::unique_ptr<ObservableObject<float>> cameraSpeed,
        std::unique_ptr<ObservableObject<float>> mouseSpeed
    );
    ml::Matrix4f const &getScenePerspectiveProjectionMatrix() const noexcept;
    ml::Matrix4f const &getViewTransformation() const noexcept;

    void handleInput(CameraInput::Cases const &cameraInput, float dt) noexcept;

private:
    ml::Vector3f origin;
    ml::Vector3f lookAt = {0, 0, -1};
    std::unique_ptr<ObservableObject<float>> fov_;
    std::unique_ptr<ObservableObject<std::pair<size_t, size_t>>> windowSize_;
    std::unique_ptr<ObservableObject<RendererType>> rendererType_;
    std::unique_ptr<ObservableObject<float>> cameraSpeed_;
    std::unique_ptr<ObservableObject<float>> mouseSpeed_;
    ml::Matrix4f viewMatrix;
    ml::Matrix4f perspectiveProjectionMatrix;
    Connections connections;

    float rotate_about_x = 0, rotate_about_y = 0; // x and y
    std::optional<int32_t> last_x, last_y;
};