#pragma once

#include <cassert>

#include "GL/glew.h"

#include "Core.hpp"
#include "Attributes.hpp"

namespace gl {
    template<typename Attr>
    void bindAttributes(Attr const &test);

    template<>
    void inline bindAttributes<attributes::FloatAttr>(attributes::FloatAttr const &test) {
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
    void inline bindAttributes<attributes::Vec2>(attributes::Vec2 const &test) {
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
    void inline bindAttributes<attributes::Vec3>(attributes::Vec3 const &test) {
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
    void inline bindAttributes<attributes::Vec4>(attributes::Vec4 const &test) {
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
    void inline bindAttributes<attributes::PositionWithColor>(attributes::PositionWithColor const &test) {
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
    void inline bindAttributes<attributes::PositionWithTex>(attributes::PositionWithTex const &test) {
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
    void inline bindAttributes<attributes::MaterialVertex>(attributes::MaterialVertex const &test) {
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

    template<>
    void inline bindAttributes<attributes::Cases>(attributes::Cases const &test) {
        std::visit(overload {
            [&](attributes::Mat4 const &) { throw std::logic_error("attributes::iColor as vertex attributes is nonsense"); },
            [&](attributes::iColor const &) { throw std::logic_error("attributes::iColor as vertex attributes is nonsense"); },
            [&](attributes::Transforms const &) { throw std::logic_error("attributes::Transforms as vertex attributes is nonsense"); },
            [&](attributes::IntegerAttr const &) { throw std::logic_error("attributes::IntegerAttr as vertex attributes is nonsense"); },
            [&]<typename T>(T const &attr) { bindAttributes<T>(attr); }
        }, test);
    };
}