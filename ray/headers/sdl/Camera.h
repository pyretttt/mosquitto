#pragma once

#include <utility>

#include "MathUtils.h"
#include "ReactivePrimitives.h"

namespace sdl {

class Camera final {
public:
    Camera() = delete;
    Camera(Camera &&other) = delete;
    Camera(Camera const &other) = delete;
    Camera operator=(Camera &&other) = delete;
    Camera operator=(Camera const &other) = delete;

    explicit Camera(
        ObservableObject<float> const &fov,
        ObservableObject<std::pair<size_t, size_t>> const &windowSize
    );
    ml::Matrix4f const &getScenePerspectiveProjectionMatrix() const noexcept;
    ml::Matrix4f const &getCameraTransformation() const noexcept;

private:
    float fov;
    float aspectRatio;
    ml::Matrix4f transformation;
    ml::Matrix4f perspectiveProjectionMatrix;
    Connections connections;
};
}