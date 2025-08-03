#pragma once

#include <filesystem>
#include <unordered_map>

#include "scene/Identifiers.hpp"
#include "scene/Attachment.hpp"
#include "Attributes.hpp"

namespace scene {

struct Scene {
    Scene static assimpImport(std::filesystem::path path);

    Scene(
        std::unordered_map<ID, NodePtr> nodes,
        std::unordered_map<ID, MaterialPtr> materials,
        std::unordered_map<TexturePath, TexturePtr> textures,
        std::unordered_map<ID, AttachmentCases> attachments
    );
    
    std::unordered_map<ID, NodePtr> nodes;
    std::unordered_map<ID, AttachmentCases> attachments;
    std::unordered_map<ID, MaterialPtr> materials;
    std::unordered_map<TexturePath, TexturePtr> textures;
};
}