#pragma once

#include <variant>

#include "MathUtils.hpp"

namespace attributes {
    struct FloatAttr {
        float val;
    };

    template<size_t size>
    struct Vec {
        float val[size];
    };

    struct iColor {
        uint32_t val;
    };

    using Vec2 = Vec<2>;
    using Vec3 = Vec<3>;
    using Vec4 = Vec<4>;

    struct PositionWithColor {
        Vec3 position;
        Vec4 color;
    };

    struct PositionWithTex {
        Vec3 position;
        Vec2 tex;
    };

    using Mat4 = ml::Matrix4f;

    struct Transforms {
        Mat4 worldMatrix;
        Mat4 viewMatrix;
        Mat4 projectionMatrix;

        Transforms(
            Mat4 worldMatrix,
            Mat4 viewMatrix,
            Mat4 projectionMatrix
        ) 
            : worldMatrix(worldMatrix)
            , viewMatrix(viewMatrix)
            , projectionMatrix(projectionMatrix) {}
    };

    struct PosNormalTex {
        Vec3 position;
        Vec3 normal;
        Vec2 tex;
    };

    using UniformCases = std::variant<
        FloatAttr,
        Vec2,
        Vec3,
        Vec4,
        iColor,
        Mat4
    >;
}