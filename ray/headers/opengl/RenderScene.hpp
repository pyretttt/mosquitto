#pragma once

#include "scene/Scene.hpp"
#include "scene/Identifiers.hpp"
#include "opengl/RenderPipeline.hpp"
#include "opengl/glCommon.hpp"
#include "opengl/glFramebuffers.hpp"
#include "opengl/Shading.hpp"
#include "Attributes.hpp"

namespace gl {
struct RenderPipelineInfo {
    size_t nodeId;
    gl::RenderPipeline<> renderPipeline;
};

struct RenderSceneStageAction {
    std::function<void ()> preRender;
    std::function<void ()> postRender;
};

struct RenderScene {
    RenderScene(
        scene::ScenePtr scene,
        gl::PipelineConfiguration configuration,
        FramebufferInfo framebuffer,
        gl::Shading shading
    );

    void render() const;
    void prepare();

    scene::ScenePtr scene;
    gl::Shading shading;
    gl::PipelineConfiguration configuration;
    FramebufferInfo framebufferInfo;
    RenderSceneStageAction actions;
};
}