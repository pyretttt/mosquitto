#include <iostream>

#include "sdl/Camera.h"

sdl::Camera::Camera(
    ObservableObject<float> fov,
    ObservableObject<std::pair<size_t, size_t>> windowSize
) : fov(fov.value()), aspectRatio(static_cast<float>(windowSize.value().first) / windowSize.value().second) {
    transformation = ml::cameraMatrix(
        {15, 0, 0},
        {15, 0, -1}
    );
    perspectiveProjectionMatrix = ml::perspectiveProjectionMatrix(this->fov, this->aspectRatio, true, 0.1, 1000);
    disposePool.push_back(
        fov.addObserver([this](float value) {
            this->fov = value;
            this->perspectiveProjectionMatrix = ml::perspectiveProjectionMatrix(value, this->aspectRatio, true, 0.1, 1000);
        })
    );
    disposePool.push_back(
        windowSize.addObserver([this](std::pair<size_t, size_t> value) {
            this->aspectRatio = static_cast<float>(value.first) / value.second;
            this->perspectiveProjectionMatrix = ml::perspectiveProjectionMatrix(this->fov, this->aspectRatio, true, 0.1, 1000);
        })
    );
}

ml::Matrix4f const &sdl::Camera::getCameraTransformation() const noexcept {
    return transformation;
}

ml::Matrix4f const &sdl::Camera::getScenePerspectiveProjectionMatrix() const noexcept {
    return perspectiveProjectionMatrix;
}