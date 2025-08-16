#pragma once

#include <cassert>

#include "GL/glew.h"

#include "Core.hpp"
#include "Attributes.hpp"

namespace gl {
    template<typename Attr>
    void bindAttributes(Attr const &test, unsigned long stride = sizeof(Attr));

    template<>
    void inline bindAttributes<attributes::FloatAttr>(attributes::FloatAttr const &test, unsigned long stride) {
        glVertexAttribPointer(
            0,
            1,
            GL_FLOAT, 
            GL_FALSE, 
            stride,
            (void *)0
        );
        glEnableVertexAttribArray(0);
    }

    template<>
    void inline bindAttributes<attributes::Vec2>(attributes::Vec2 const &test, unsigned long stride) {
        glVertexAttribPointer(
            0,
            2,
            GL_FLOAT, 
            GL_FALSE, 
            stride,
            (void *)0
        );
        glEnableVertexAttribArray(0);
    }

    template<>
    void inline bindAttributes<attributes::Vec3>(attributes::Vec3 const &test, unsigned long stride) {
        glVertexAttribPointer(
            0,
            3,
            GL_FLOAT, 
            GL_FALSE, 
            stride,
            (void *)0
        );
        glEnableVertexAttribArray(0);
    }

    template<>
    void inline bindAttributes<attributes::Vec4>(attributes::Vec4 const &test, unsigned long stride) {
        glVertexAttribPointer(
            0,
            4,
            GL_FLOAT, 
            GL_FALSE, 
            stride,
            (void *)0
        );
        glEnableVertexAttribArray(0);
    }

    template<>
    void inline bindAttributes<attributes::PositionWithColor>(attributes::PositionWithColor const &test, unsigned long stride) {
        glVertexAttribPointer(
            0,
            3,
            GL_FLOAT, 
            GL_FALSE, 
            stride,
            (void *)0
        );
        glVertexAttribPointer(
            1,
            4,
            GL_FLOAT, 
            GL_FALSE, 
            stride,
            (void *)offsetof(attributes::PositionWithColor, color)
        );
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
    }

    template<>
    void inline bindAttributes<attributes::PositionWithTex>(attributes::PositionWithTex const &test, unsigned long stride) {
        glVertexAttribPointer(
            0,
            3,
            GL_FLOAT, 
            GL_FALSE,
            stride,
            (void *)0
        );

        glVertexAttribPointer(
            1,
            2,
            GL_FLOAT, 
            GL_FALSE, 
            stride,
            (void *)offsetof(attributes::PositionWithTex, tex)
        );
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
    }

    template<>
    void inline bindAttributes<attributes::MaterialVertex>(attributes::MaterialVertex const &test, unsigned long stride) {
        glVertexAttribPointer(
            0,
            3,
            GL_FLOAT, 
            GL_FALSE, 
            stride,
            (void *)0
        );

        glVertexAttribPointer(
            1,
            3,
            GL_FLOAT, 
            GL_FALSE, 
            stride,
            (void *)offsetof(attributes::MaterialVertex, normal)
        );

        glVertexAttribPointer(
            2,
            2,
            GL_FLOAT, 
            GL_FALSE, 
            stride,
            (void *)offsetof(attributes::MaterialVertex, tex)
        );
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glEnableVertexAttribArray(2);
    }

    template<>
    void inline bindAttributes<attributes::Cases>(attributes::Cases const &test, unsigned long stride) {
        std::visit(overload {
            [&](attributes::iColor const &) { throw std::logic_error("attributes::iColor as vertex attributes is nonsense"); },
            [&](attributes::IntegerAttr const &) { throw std::logic_error("attributes::IntegerAttr as vertex attributes is nonsense"); },
            [&]<typename T>(T const &attr) { bindAttributes<T>(attr, stride); }
        }, test);
    };
}