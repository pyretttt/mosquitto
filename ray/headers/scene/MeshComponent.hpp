#pragma once

#include "scene/Component.hpp"
#include "Attributes.hpp"
#include "scene/Attachment.hpp"

namespace scene {
    template <typename Attribute, typename Attachment>
    using MeshPtr = std::shared_ptr<Mesh<Attribute, Attachment>>;

    template<typename Attribute = attributes::Cases, typename Attachment = AttachmentCases>
    using MeshComponent = ContainerComponent<std::vector<MeshPtr<Attribute, Attachment>>;
}