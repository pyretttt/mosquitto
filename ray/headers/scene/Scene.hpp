#pragma once

#include <filesystem>
#include <unordered_map>

#include "scene/Identifiers.hpp"
#include "Attributes.hpp"

namespace scene {

struct Scene {
    Scene static assimpImport(std::filesystem::path path);

    Scene(
        std::unordered_map<NodeId, NodePtr> nodes,
        std::unordered_map<MaterialId, MaterialPtr> materials,
        std::unordered_map<TexturePath, TexturePtr> textures
    );
    
    std::unordered_map<NodeId, NodePtr> nodes;
    std::unordered_map<MaterialId, MaterialPtr> materials;

private:
    std::unordered_map<TexturePath, TexturePtr> textures;
};
}