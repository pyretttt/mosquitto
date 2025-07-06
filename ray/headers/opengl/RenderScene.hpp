#pragma once

#include "scene/Scene.hpp"
#include "scene/Identifiers.hpp"
#include "opengl/RenderObject.hpp"
#include "opengl/glCommon.hpp"
#include "opengl/Shader.hpp"
#include "Attributes.hpp"

namespace gl {
struct RenderObjectInfo {
    size_t nodeId;
    gl::RenderObject<attributes::AssimpVertex> renderObject;
};

struct RenderScene {
    RenderScene(
        scene::ScenePtr scene,
        gl::Configuration configuration,
        ShaderPtr shader,
        FramebufferInfo framebuffer
    );

    void render() const;
    void prepare();

    scene::ScenePtr scene;
    gl::Configuration configuration;
    std::vector<RenderObjectInfo> pbrs;
    std::unordered_map<scene::MaterialId, Material> materials;

    ShaderPtr shader;
    FramebufferInfo framebufferInfo;
};
}