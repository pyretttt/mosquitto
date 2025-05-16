#include <iostream>
#include <cmath>

#include "Camera.hpp"

Camera::Camera(
    std::unique_ptr<ObservableObject<float>> &&fov,
    std::unique_ptr<ObservableObject<std::pair<size_t, size_t>>> &&windowSize
) : fov_(std::move(fov)),
    windowSize_(std::move(windowSize)),
    viewMatrix(ml::cameraMatrix(origin, lookAt)),
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

void Camera::handleInput(CameraInput::Cases const &input) noexcept {
    if (auto const *value = std::get_if<CameraInput::Translate>(&input)) {
        origin.z += +value->backward - value->forward;
        origin.x += value->right - value->left;
    } else if (auto const *value = std::get_if<CameraInput::Rotate>(&input)) {
        static float const mouseSpeed = 0.01f;
        float dy = value->delta.first * mouseSpeed;
        float dx = value->delta.second * mouseSpeed;
        rotate_about_x += dy;
        rotate_about_x = std::max(std::min(rotate_about_x, ml::kPI_2), -ml::kPI_2);
        rotate_about_y += dx;
    }

    lookAt.x = sinf(rotate_about_y);
    lookAt.y = -cosf(rotate_about_y) * sinf(rotate_about_x);
    lookAt.z = -cosf(rotate_about_x) * cosf(rotate_about_y);

    viewMatrix = ml::cameraMatrix(origin, lookAt + origin);
}

ml::Matrix4f const &Camera::getViewTransformation() const noexcept {
    return viewMatrix;
}

ml::Matrix4f const &Camera::getScenePerspectiveProjectionMatrix() const noexcept {
    return perspectiveProjectionMatrix;
}