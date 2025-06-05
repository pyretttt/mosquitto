#pragma once

#include <filesystem>
#include <unordered_map>

#include "scene/Mesh.hpp"
#include "scene/Tex.hpp"
#include "scene/Node.hpp"
#include "scene/Material.hpp"
#include "Attributes.hpp"

namespace scene {
    using TexturePtr = std::shared_ptr<TexData>;

struct Scene {
    Scene static assimpImport(std::filesystem::path path);

    Scene(
        std::unordered_map<size_t, NodePtr> nodes,
        std::unordered_map<size_t, MaterialPtr> materials
    );

    std::unordered_map<size_t, NodePtr> nodes;
    std::unordered_map<size_t, TexturePtr> materials;
};
}