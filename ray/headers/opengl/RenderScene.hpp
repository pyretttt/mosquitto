#pragma once

#include "scene/Scene.hpp"
#include "opengl/RenderObject.hpp"
#include "Attributes.hpp"

namespace gl {
struct RenderObjectInfo {
    size_t nodeId;
    gl::RenderObject<attributes::AssimpVertex> renderObject;
};

struct Material {
    std::vector<gl::TexturePtr> ambient;
    std::vector<gl::TexturePtr> diffuse;
    std::vector<gl::TexturePtr> specular;
    std::vector<gl::TexturePtr> normals;
};


struct RenderScene {
    RenderScene(
        scene::ScenePtr scene,
        gl::Configuration configuration
    );

    void render() const;

    scene::ScenePtr scene;
    gl::Configuration configuration;
    std::vector<RenderObjectInfo> pbrs;
    std::unordered_map<size_t, Material> materials;
};
}