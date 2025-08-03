#pragma once

#include <variant>

#include "scene/Identifiers.hpp"
#include "scene/Component.hpp"
#include "opengl/glTexture.hpp"

namespace gl {
    struct MaterialAttachment final {
        scene::ID id;
        std::shared_ptr<gl::Material> material;

        MaterialAttachment(std::shared_ptr<gl::Material> material, scene::ID id) 
            : material(std::move(material))
            , id(id) {}
    };

    
    using AttachmentCases = std::variant<
        MaterialAttachment,
        std::monostate
    >;
}