#pragma once

#include "glm/common.hpp"

#include "MathUtils.hpp"

namespace gl {
    ml::Matrix4f glPerspectiveMatrix(
        float fovRadians,
        float aspect,
        float near,
        float far
    ) noexcept {
        return glm::perspective(fovRadians, aspect, near, far);
    }

    template <typename Vector>
    auto inline getPtr(Vector const &vec) -> decltype(glm::value_ptr(vec)) {
        return glm::value_ptr(vec);
    }
}