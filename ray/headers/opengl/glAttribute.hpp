#pragma once

#include <cassert>

#include "GL/glew.h"

#include "Core.hpp"
#include "Attributes.hpp"

namespace gl {
    template<typename Attr>
    void bindAttributes(Attr const &test);

    template<>
    void bindAttributes(attributes::Cases const &test) {
        std::visit(overload {
            [&](auto const &attr) { bindAttributes(attr); }
        });
    };

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
}