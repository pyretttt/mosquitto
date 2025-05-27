#pragma once

#include <filesystem>

#include "scene/MeshNode.hpp"
#include "Attributes.hpp"

namespace scene {
struct Scene {
    using Mesh = MeshNode<attributes::AssimpVertex>;
    using MeshPtr = std::shared_ptr<MeshNode<attributes::AssimpVertex>>;
    Scene(
        std::filesystem::path path,
        std::vector<Mesh> meshes
    );

    Scene static assimpImport(std::filesystem::path path);

    std::vector<MeshPtr> meshes;
};
}