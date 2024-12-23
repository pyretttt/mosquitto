#pragma once

#include <utility>

#include "MathUtils.hpp"
#include "ReactivePrimitives.hpp"

namespace sdl {

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
    ml::Matrix4f const &getCameraTransformation() const noexcept;

private:
    std::unique_ptr<ObservableObject<float>> fov_;
    std::unique_ptr<ObservableObject<std::pair<size_t, size_t>>> windowSize_;
    ml::Matrix4f transformation;
    ml::Matrix4f perspectiveProjectionMatrix;
    Connections connections;
};
} // namespace sdl