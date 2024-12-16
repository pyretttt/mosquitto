#include "sdl/Camera.h"

sdl::Camera::Camera(
    rpp::subjects::behavior_subject<float> fov,
    rpp::subjects::behavior_subject<float> aspectRatio
) : fov(fov.get_value()), aspectRatio(aspectRatio.get_value()) {
    
}

ml::Matrix4f const &sdl::Camera::getCameraTransformation() const {
    return transformation;
}

ml::Matrix4f const &sdl::Camera::getScenePerspectiveProjectionMatrix() const {
    return ml::eye<4>();
}