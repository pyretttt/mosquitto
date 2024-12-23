#include <iostream>

#include "sdl/Camera.hpp"

sdl::Camera::Camera(
    std::unique_ptr<ObservableObject<float>> &&fov,
    std::unique_ptr<ObservableObject<std::pair<size_t, size_t>>> &&windowSize
) : fov_(std::move(fov)),
    windowSize_(std::move(windowSize)),
    transformation(ml::cameraMatrix({0, 0, 0}, {0, 0, -1})),
    perspectiveProjectionMatrix(
        ml::perspectiveProjectionMatrix(
            fov_->value(),
            static_cast<float>(windowSize_->value().first) / windowSize_->value().second,
            true,
            0.1,
            1000
        )
    ) {

    connections.push_back(
        fov_->subscribe([this](float value) {
            auto windowSize = this->windowSize_->value();
            auto aspectRatio = static_cast<float>(windowSize.first) / windowSize.second;
            this->perspectiveProjectionMatrix = ml::perspectiveProjectionMatrix(value, aspectRatio, true, 0.1, 1000);
        })
    );
    connections.push_back(
        windowSize_->subscribe([this](std::pair<size_t, size_t> value) {
            std::cout << this->windowSize_->value().first << " " << this->windowSize_->value().second << std::endl;
            auto newAspectRatio = static_cast<float>(value.first) / value.second;
            this->perspectiveProjectionMatrix = ml::perspectiveProjectionMatrix(this->fov_->value(), newAspectRatio, true, 0.1, 1000);
        })
    );
}

ml::Matrix4f const &sdl::Camera::getCameraTransformation() const noexcept {
    return transformation;
}

ml::Matrix4f const &sdl::Camera::getScenePerspectiveProjectionMatrix() const noexcept {
    return perspectiveProjectionMatrix;
}