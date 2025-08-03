#pragma once

#include "scene/Scene.hpp"
#include "scene/Identifiers.hpp"
#include "opengl/RenderPipeline.hpp"
#include "opengl/glCommon.hpp"
#include "opengl/glFramebuffers.hpp"
#include "opengl/Shader.hpp"
#include "Attributes.hpp"

namespace gl {
struct RenderPipelineInfo {
    size_t nodeId;
    gl::RenderPipeline<> renderPipeline;
};

struct RenderScene {
    RenderScene(
        scene::ScenePtr scene,
        gl::PipelineConfiguration configuration,
        FramebufferInfo framebuffer
    );

    void render() const;
    void prepare();

    scene::ScenePtr scene;
    gl::PipelineConfiguration configuration;
    FramebufferInfo framebufferInfo;
};
}