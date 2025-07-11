#pragma once

#include <variant>

#include "opengl/glTexture.hpp"
#include "scene/Attachment.hpp"

namespace gl {

    struct Attachment final {
        scene::CommonAttachment commonAttachment;
        std::shared_ptr<gl::Material> material;

        Attachment(scene::CommonAttachment commonAttachment);
    };

using AttachmentCases = std::variant<
    scene::CommonAttachment,
    std::monostate
>;
}