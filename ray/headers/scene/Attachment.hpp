#pragma once

#include <variant>

#include "scene/Material.hpp"
#include "scene/Identifiers.hpp"

namespace scene {
struct MaterialAttachment final {
    MaterialId id;
    MaterialPtr material;
};

using AttachmentCases = std::variant<
    MaterialAttachment, 
    std::monostate
>;
}