#pragma once

#include <variant>

#include "opengl/glTexture.hpp"
#include "scene/Component.hpp"

namespace gl {
    struct Attachment final {
        std::shared_ptr<gl::Material> material;

        Attachment(std::shared_ptr<gl::Material> material) 
            : material(std::move(material)) {}
    };

    using AttachmentComponent = scene::ContainerComponent<Attachment>;

    using AttachmentCases = std::variant<
        Attachment,
        std::monostate
    >;
}