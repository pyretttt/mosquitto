#include <iostream>
#include <cmath>

#include "glm/common.hpp"

#include "Camera.hpp"
#include "RendererBase.hpp"
#include "glMath.hpp"

namespace {
    constexpr float near = 0.1f;
    constexpr float far = 1000.f;

    ml::Matrix4f getPerspectiveMatrix(
        RendererType renderer,
        float fov,
        float aspect
    ) noexcept {
        switch (renderer) {
        case RendererType::CPU:
            return ml::perspectiveProjectionMatrix(fov, aspect, true, near, far);
            break;
        case RendererType::OpenGL:
            return gl::glPerspectiveMatrix(fov, aspect, near, far);
            break;
        }
    }
}

Camera::Camera(
    std::unique_ptr<ObservableObject<float>> &&fov,
    std::unique_ptr<ObservableObject<std::pair<size_t, size_t>>> &&windowSize,
    std::unique_ptr<ObservableObject<RendererType>> &&rendererType
) 
    : fov_(std::move(fov))
    , windowSize_(std::move(windowSize))
    , viewMatrix(ml::cameraMatrix(origin, lookAt))
    , rendererType_(std::move(rendererType))
    , perspectiveProjectionMatrix(
        getPerspectiveMatrix(
            rendererType->value(),
            ml::toRadians(fov_->value()), 
            static_cast<float>(windowSize_->value().first) / windowSize_->value().second 
        )
    ) {

    connections.push_back(
        fov_->subscribe([this](float value) {
            auto windowSize = this->windowSize_->value();
            auto aspectRatio = static_cast<float>(windowSize.first) / windowSize.second;
            // this->perspectiveProjectionMatrix = ml::perspectiveProjectionMatrix(value, aspectRatio, true, near, far);
            getPerspectiveMatrix(
                this->rendererType_->value(),
                ml::toRadians(fov_->value()), 
                aspectRatio
            );
            this->perspectiveProjectionMatrix = glm::perspective(glm::radians(45.f), aspectRatio, near, far);
        })
    );
    connections.push_back(
        windowSize_->subscribe([this](std::pair<size_t, size_t> value) {
            std::cout << this->windowSize_->value().first << " " << this->windowSize_->value().second << std::endl;
            auto newAspectRatio = static_cast<float>(value.first) / value.second;
            // this->perspectiveProjectionMatrix = ml::perspectiveProjectionMatrix(this->fov_->value(), newAspectRatio, true, near, far);
            this->perspectiveProjectionMatrix = glm::perspective(glm::radians(45.f), newAspectRatio, near, far);
        })
    );

    connections.push_back(
        rendererType_->subscribe([this](RendererType renderer) {
            switch (renderer) {

            }
        });
    )
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