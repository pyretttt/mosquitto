#pragma once

#include "scene/Scene.hpp"
#include "scene/Identifiers.hpp"
#include "opengl/RenderPipeline.hpp"
#include "opengl/glCommon.hpp"
#include "opengl/Shader.hpp"
#include "Attributes.hpp"

namespace gl {
struct RenderPipelineInfo {
    size_t nodeId;
    gl::RenderPipeline<attributes::MaterialVertex> RenderPipeline;
};

struct RenderScene {
    RenderScene(
        scene::ScenePtr scene,
        gl::PipelineConfiguration configuration,
        ShaderPtr shader,
        FramebufferInfo framebuffer
    );

    void render() const;
    void prepare();

    scene::ScenePtr scene;
    gl::PipelineConfiguration configuration;
    std::vector<RenderPipelineInfo> pbrs;
    std::unordered_map<scene::MaterialId, Material> materials;

    ShaderPtr shader;
    FramebufferInfo framebufferInfo;
};
}