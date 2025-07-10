#pragma once

#include <cassert>

#include "GL/glew.h"

#include "Attributes.hpp"

namespace gl {
    template<typename Attr>
    void bindAttributes();

    template<>
    void inline bindAttributes<attributes::FloatAttr>() {
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
    void inline bindAttributes<attributes::Vec2>() {
        glVertexAttribPointer(
            0,
            2,
            GL_FLOAT, 
            GL_FALSE, 
            sizeof(decltype(attributes::Vec2::val)),
            (void *)0
        );
        glEnableVertexAttribArray(0);
    }

    template<>
    void inline bindAttributes<attributes::Vec3>() {
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
    void inline bindAttributes<attributes::Vec4>() {
        glVertexAttribPointer(
            0,
            4,
            GL_FLOAT, 
            GL_FALSE, 
            sizeof(decltype(attributes::Vec4::val)),
            (void *)0
        );
        glEnableVertexAttribArray(0);
    }

    template<>
    void inline bindAttributes<attributes::PositionWithColor>() {
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

    template<>
    void inline bindAttributes<attributes::PositionWithTex>() {
        glVertexAttribPointer(
            0,
            3,
            GL_FLOAT, 
            GL_FALSE, 
            sizeof(attributes::PositionWithTex),
            (void *)0
        );

        glVertexAttribPointer(
            1,
            2,
            GL_FLOAT, 
            GL_FALSE, 
            sizeof(attributes::PositionWithTex),
            (void *)offsetof(attributes::PositionWithTex, tex)
        );
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
    }

    template<>
    void inline bindAttributes<attributes::MaterialVertex>() {
        glVertexAttribPointer(
            0,
            3,
            GL_FLOAT, 
            GL_FALSE, 
            sizeof(attributes::MaterialVertex),
            (void *)0
        );

        glVertexAttribPointer(
            1,
            3,
            GL_FLOAT, 
            GL_FALSE, 
            sizeof(attributes::MaterialVertex),
            (void *)offsetof(attributes::MaterialVertex, normal)
        );

        glVertexAttribPointer(
            2,
            2,
            GL_FLOAT, 
            GL_FALSE, 
            sizeof(attributes::MaterialVertex),
            (void *)offsetof(attributes::MaterialVertex, tex)
        );
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glEnableVertexAttribArray(2);
    }
}