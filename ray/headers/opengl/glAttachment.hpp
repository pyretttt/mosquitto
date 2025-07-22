#pragma once

#include <variant>

#include "scene/Identifiers.hpp"
#include "scene/Component.hpp"
#include "opengl/glTexture.hpp"

namespace gl {
    struct MaterialAttachment final {
        scene::MaterialId id;
        std::shared_ptr<gl::Material> material;

        MaterialAttachment(std::shared_ptr<gl::Material> material, scene::MaterialId id) 
            : material(std::move(material))
            , id(id) {}
    };

    
    using AttachmentCases = std::variant<
        MaterialAttachment,
        std::monostate
    >;
}