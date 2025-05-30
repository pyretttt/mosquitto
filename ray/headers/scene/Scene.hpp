#pragma once

#include <filesystem>
#include <unordered_map>

#include "scene/MeshNode.hpp"
#include "Attributes.hpp"

namespace scene {
struct Scene {
    using Mesh = SceneNode<attributes::AssimpVertex>;
    using MeshPtr = std::shared_ptr<SceneNode<attributes::AssimpVertex>>;
    Scene(
        std::filesystem::path path,
        std::vector<Mesh> meshes
    );

    Scene static assimpImport(std::filesystem::path path);

    std::unordered_map<size_t, MeshPtr> meshes;
};
}