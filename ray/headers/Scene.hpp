#pragma once

#include <filesystem>

#include "opengl/MeshNode.hpp"
#include "Attributes.hpp"

namespace system {
struct Scene {
    using Mesh = gl::MeshNode<attributes::AssimpVertex>;
    using MeshPtr = std::shared_ptr<gl::MeshNode<attributes::AssimpVertex>>;
    Scene(
        std::filesystem::path path,
        std::vector<Mesh> meshes
    );

    Scene static assimpImport(std::filesystem::path path);

    std::vector<MeshPtr> meshes;
};
}