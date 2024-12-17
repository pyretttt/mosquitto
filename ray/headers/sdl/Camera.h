#pragma once

#include <utility>

#include "MathUtils.h"
#include "ReactivePrimitives.h"

namespace sdl {
class Camera final {
public:
    Camera() = delete;
    Camera(
        ObservableObject<float> fov,
        ObservableObject<std::pair<size_t, size_t>> windowSize
    );
    ml::Matrix4f const &getScenePerspectiveProjectionMatrix() const;
    ml::Matrix4f const &getCameraTransformation() const;

private:
    ml::Matrix4f transformation;
    ml::Matrix4f perspectiveProjectionMatrix;
    float fov;
    float aspectRatio;
    DisposePool disposePool;
};
}