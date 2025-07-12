#pragma once

#include <variant>

#include "opengl/glTexture.hpp"
#include "scene/Attachment.hpp"

namespace gl {

    struct Attachment final {
        scene::CommonAttachment commonAttachment;
        std::shared_ptr<gl::Material> material;

        Attachment(
            scene::CommonAttachment commonAttachment,
            std::shared_ptr<gl::Material> material
        ) 
            : commonAttachment(commonAttachment)
            , material(std::move(material)) {
        }
    };

    using AttachmentCases = std::variant<
        Attachment,
        std::monostate
    >;
}