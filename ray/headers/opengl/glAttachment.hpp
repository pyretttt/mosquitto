#pragma once

#include <variant>

#include "opengl/glTexture.hpp"
#include "scene/Identifiers.hpp"
#include "scene/Component.hpp"

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