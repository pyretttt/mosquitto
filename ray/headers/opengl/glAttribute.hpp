#pragma once

#include <cassert>

#include "GL/glew.h"

#include "Attributes.hpp"

namespace gl {
    template<typename Attr>
    void bindAttributes();

    template<>
    void bindAttributes<attributes::FloatAttr>() {
        glVertexAttribPointer(
            0,
            1,
            GL_FLOAT, 
            GL_FALSE, 
            sizeof(decltype(attributes::FloatAttr::val)),
            (void *)0
        );
        glEnableVertexAttribArray(0);
    }

    template<>
    void bindAttributes<attributes::Vec3>() {
        glVertexAttribPointer(
            0,
            3,
            GL_FLOAT, 
            GL_FALSE, 
            sizeof(decltype(attributes::Vec3::val)),
            (void *)0
        );
        glEnableVertexAttribArray(0);
    }

    template<>
    void bindAttributes<attributes::PositionWithColor>() {
        glVertexAttribPointer(
            0,
            3,
            GL_FLOAT, 
            GL_FALSE, 
            sizeof(attributes::PositionWithColor),
            (void *)0
        );
        glVertexAttribPointer(
            1,
            4,
            GL_FLOAT, 
            GL_FALSE, 
            sizeof(attributes::PositionWithColor),
            (void *)offsetof(attributes::PositionWithColor, color)
        );
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
    }

}