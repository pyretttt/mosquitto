#pragma once

#include <variant>

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

    struct PositionWithText {
        Vec3 position;
        Vec2 tex;
    };

    using BaseCases = std::variant<
        FloatAttr,
        Vec2,
        Vec3,
        Vec4,
        iColor
    >;
}