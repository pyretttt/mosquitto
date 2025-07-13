#include <iostream>
#include <cmath>

#include "glm/common.hpp"

#include "Camera.hpp"
#include "RendererBase.hpp"
#include "opengl/glMath.hpp"
#include "Core.hpp"

namespace {
    constexpr float maxZoom = 100.f;
    constexpr float near = 0.1f;
    constexpr float far = 100.f;

    ml::Matrix4f getPerspectiveMatrix(
        RendererType renderer,
        float fov,
        float aspect
    ) noexcept {
        switch (renderer) {
        case RendererType::CPU:
            return ml::perspectiveProjectionMatrix(fov, aspect, true, near, far);
        case RendererType::OpenGL:
            return gl::glPerspectiveMatrix(fov, aspect, near, far);
        }
    }
}

Camera::Camera(
    std::unique_ptr<ObservableObject<float>> fov,
    std::unique_ptr<ObservableObject<std::pair<size_t, size_t>>> windowSize,
    std::unique_ptr<ObservableObject<RendererType>> rendererType,
    std::unique_ptr<ObservableObject<float>> cameraSpeed,
    std::unique_ptr<ObservableObject<float>> mouseSpeed
) 
    : fov_(std::move(fov))
    , windowSize_(std::move(windowSize))
    , viewMatrix(ml::cameraMatrix(origin, lookAt))
    , rendererType_(std::move(rendererType))
    , cameraSpeed_(std::move(cameraSpeed))
    , mouseSpeed_(std::move(mouseSpeed))
    , perspectiveProjectionMatrix(
        getPerspectiveMatrix(
            rendererType_->value(),
            fov_->value(), 
            static_cast<float>(windowSize_->value().first) / windowSize_->value().second 
        )
    ) {

    connections.push_back(
        fov_->subscribe([this](float value) {
            auto windowSize = this->windowSize_->value();
            auto aspectRatio = static_cast<float>(windowSize.first) / windowSize.second;
            this->perspectiveProjectionMatrix = getPerspectiveMatrix(
                this->rendererType_->value(),
                this->fov_->value(),
                aspectRatio
            );
        })
    );
    connections.push_back(
        windowSize_->subscribe([this](std::pair<size_t, size_t> value) {
            std::cout << value.first << " " << value.second << std::endl;
            auto newAspectRatio = static_cast<float>(value.first) / value.second;
            this->perspectiveProjectionMatrix = getPerspectiveMatrix(
                this->rendererType_->value(),
                this->fov_->value(),
                newAspectRatio
            );
        })
    );

    connections.push_back(
        rendererType_->subscribe([this](RendererType renderer) {
            auto windowSize = this->windowSize_->value();
            auto aspectRatio = static_cast<float>(windowSize.first) / windowSize.second;
            this->perspectiveProjectionMatrix = getPerspectiveMatrix(
                renderer,
                this->fov_->value(),
                aspectRatio
            );
        })
    );
}

void Camera::handleInput(CameraInput::Cases const &input, float dt) noexcept {
    std::visit(overload {
        [&](CameraInput::Translate const &value) {
            auto forward = static_cast<int>(value.backward) - static_cast<int>(value.forward);
            auto side = static_cast<int>(value.right) - static_cast<int>(value.left);
            auto dist = cameraSpeed_->value() * dt;
            origin.z += forward * dist;
            origin.x += side * dist;
        },
        [&](CameraInput::Rotate const &value) {
            auto const mouseSpeed = mouseSpeed_->value();
            float dy = value.delta.first * mouseSpeed;
            float dx = value.delta.second * mouseSpeed;
            rotate_about_x += dy;
            rotate_about_x = std::max(std::min(rotate_about_x, ml::kPI_2), -ml::kPI_2);
            rotate_about_y += dx;
        }
    }, input);

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

ml::Vector3f const &Camera::getOrigin() const noexcept {
    return origin;
}