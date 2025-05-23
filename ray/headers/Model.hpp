#pragma once

#include <filesystem>

#include "opengl/MeshNode.hpp"
#include "Attributes.hpp"

struct Model {
    Model(std::filesystem::path path);

    Model static assimpImport(std::filesystem::path path);

    std::vector<gl::MeshNode<attributes::PositionWithTex>> meshes;
};