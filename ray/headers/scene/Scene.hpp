#pragma once

#include <filesystem>
#include <unordered_map>

#include "scene/SceneNode.hpp"
#include "scene/Tex.hpp"
#include "Attributes.hpp"

namespace scene {
struct Scene {
    using Node = SceneNode<attributes::AssimpVertex>;
    using NodePtr = std::shared_ptr<SceneNode<attributes::AssimpVertex>>;
    // Scene(
    //     std::filesystem::path path,
    //     std::vector<Mesh> meshes
    // );

    Scene static assimpImport(std::filesystem::path path);

    std::unordered_map<size_t, MeshPtr> meshes;
    std::unordered_map<size_t, TexData> textures;
};
}