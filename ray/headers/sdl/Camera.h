#pragma once

#include "MathUtils.h"
#include "rpp/rpp.hpp"

namespace sdl {
class Camera {
public:
    Camera() = default;
    Camera(
        rpp::subjects::behavior_subject<float> fov,
        rpp::subjects::behavior_subject<float> aspectRatio
    );
    ml::Matrix4f const &getScenePerspectiveProjectionMatrix() const;
    ml::Matrix4f const &getCameraTransformation() const;

private:
    ml::Matrix4f transformation;
    ml::Matrix4f perspectiveProjectionMatrix;
    float fov;
    float aspectRatio;
};
}