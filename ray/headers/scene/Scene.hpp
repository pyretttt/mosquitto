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
    
    bool prepare() const;

    std::unordered_map<size_t, NodePtr> nodes;
    std::unordered_map<size_t, MaterialPtr> materials;
};

using ScenePtr = std::shared_ptr<Scene>;
}