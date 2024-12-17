#include "sdl/Camera.h"

sdl::Camera::Camera(
    ObservableObject<float> fov,
    ObservableObject<std::pair<size_t, size_t>> windowSize
) : fov(fov.value()), aspectRatio(static_cast<float>(windowSize.value().first) / windowSize.value().second) {
    disposePool.push_back(
        fov.addObserver([this](float value){
            this->fov = value;
        })
    );
    disposePool.push_back(
        windowSize.addObserver([this](std::pair<size_t, size_t> value) {
            this->aspectRatio = static_cast<float>(value.first) / value.second;
        })
    );
}

ml::Matrix4f const &sdl::Camera::getCameraTransformation() const {
    return transformation;
}

ml::Matrix4f const &sdl::Camera::getScenePerspectiveProjectionMatrix() const {
    return ml::eye<4>();
}