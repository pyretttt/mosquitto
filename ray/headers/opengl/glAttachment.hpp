#pragma once

#include <variant>

#include "opengl/glTexture.hpp"

namespace gl {
using Attachment = std::variant<
    Material,
    std::monostate
>;
}