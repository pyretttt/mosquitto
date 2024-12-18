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

    Camera(
        ObservableObject<float> fov,
        ObservableObject<std::pair<size_t, size_t>> windowSize
    );
    ml::Matrix4f const &getScenePerspectiveProjectionMatrix() const noexcept;
    ml::Matrix4f const &getCameraTransformation() const noexcept;

private:
    ml::Matrix4f transformation;
    ml::Matrix4f perspectiveProjectionMatrix;
    float fov;
    float aspectRatio;
    DisposePool disposePool;
};
}