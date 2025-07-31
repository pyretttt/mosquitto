#pragma once

#include <filesystem>
#include <optional>
#include <array>
#include <type_traits>
#include <algorithm>

#include <GL/glew.h>

#include "LoadTextFile.hpp"
#include "glAttribute.hpp"
#include "Core.hpp"
#include "opengl/glCommon.hpp"
#include "opengl/glTexture.hpp"
#include "opengl/glAttachment.hpp"
#include "scene/Mesh.hpp"
#include "opengl/Shader.hpp"

namespace gl {

using EBO = std::vector<unsigned int>;

struct PipelineConfiguration {
    struct PolygonMode {
        GLenum face = GL_FRONT;
        GLenum mode = GL_FILL;
    };

    struct Stencil {
        GLenum stencilFunc = GL_ALWAYS;
        GLint ref = 1;
        GLuint mask = 0xFF;
        GLenum stencilFailOp = GL_KEEP;
        GLenum zFailOp = GL_KEEP;
        GLenum zPassOp = GL_REPLACE;
    };

    PolygonMode polygonMode;
    Stencil stencil;
};

void bindTextures(std::vector<gl::TexturePtr> const &textures);
void activateMaterial(gl::Material const &material);

template <typename Attribute = attributes::Cases>
struct RenderPipeline {
    RenderPipeline(
        PipelineConfiguration configuration,
        std::shared_ptr<scene::Mesh<Attribute, gl::AttachmentCases>> meshNode,
        ShaderPtr shader
    );

    RenderPipeline(RenderPipeline<Attribute> const &) = delete;
    RenderPipeline<Attribute>& operator=(RenderPipeline<Attribute> const&) = delete;

    RenderPipeline(RenderPipeline<Attribute> &&);
    RenderPipeline<Attribute>& operator=(RenderPipeline<Attribute>&& other);

    void prepare();
    void render() const noexcept;

    ~RenderPipeline();

    ID vbo = 0;
    ID vao = 0;
    ID ebo = 0;

    std::shared_ptr<scene::Mesh<Attribute, gl::AttachmentCases>> meshNode;
    PipelineConfiguration configuration;
    ShaderPtr shader;
};

template<typename Attribute>
RenderPipeline<Attribute>::RenderPipeline(
    PipelineConfiguration configuration,
    std::shared_ptr<scene::Mesh<Attribute, gl::AttachmentCases>> meshNode,
    ShaderPtr shader
) 
    : configuration(configuration)
    , meshNode(meshNode)
    , shader(std::move(shader)) {
}

template<typename Attribute>
RenderPipeline<Attribute>::RenderPipeline(RenderPipeline<Attribute> &&other)
    : vao(other.vao)
    , ebo(other.ebo)
    , vbo(other.vbo)
    , configuration(other.configuration)
    , meshNode(std::move(other.meshNode))
    , shader(std::move(other.shader))
{
    other.vbo = 0;
    other.ebo = 0;
    other.vao = 0;
}

template<typename Attribute>
RenderPipeline<Attribute>& RenderPipeline<Attribute>::operator=(RenderPipeline<Attribute>&& other) {
    this->configuration = other.configuration;
    this->meshNode = std::move(other.meshNode);
    this->shader = std::move(other.shader);
    this->vao = other.vao;
    this->vbo = other.vbo;
    this->ebo = other.ebo;
    other.vao = 0;
    other.ebo = 0;
    other.vbo = 0;
}

template<typename Attribute>
RenderPipeline<Attribute>::~RenderPipeline() {
    if (vbo) glDeleteBuffers(1, &vbo);
    if (ebo) glDeleteBuffers(1, &ebo);
    if (vao) glDeleteVertexArrays(1, &vao);
}

template<typename Attribute>
void RenderPipeline<Attribute>::prepare() {
    std::visit(overload {
        [&](MaterialAttachment const &attachment) {
            bindTextures(attachment.material->ambient);
            bindTextures(attachment.material->diffuse);
            bindTextures(attachment.material->specular);
            bindTextures(attachment.material->normals);
        },
        [&](std::monostate &) {}
    }, meshNode->attachment);

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(
        GL_ARRAY_BUFFER, 
        meshNode->vertexArray.size() * sizeof(Attribute),
        meshNode->vertexArray.data(), 
        GL_STATIC_DRAW
    );
    if (meshNode->vertexArray.size()) {
        bindAttributes<Attribute>(meshNode->vertexArray[0]);
    } else {
        throw "EMPTY::MESH";
    }

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER, 
        sizeof(EBO::value_type) * meshNode->vertexArrayIndices.size(),
        meshNode->vertexArrayIndices.data(),
        GL_STATIC_DRAW
    );

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

template<typename Attribute>
void RenderPipeline<Attribute>::render() const noexcept {
    std::visit(overload {
        [&](MaterialAttachment const &attachemnt) {
            activateMaterial(*attachemnt.material);
            shader->setUniform("material", *attachemnt.material);
        },
        [&](std::monostate &) {}
    }, meshNode->attachment);

    glStencilFunc(
        configuration.stencil.stencilFunc, 
        configuration.stencil.ref, 
        configuration.stencil.mask
    );
    glStencilOp(
        configuration.stencil.stencilFailOp, 
        configuration.stencil.zFailOp, 
        configuration.stencil.zPassOp
    );

    glBindVertexArray(vao);
    glPolygonMode(configuration.polygonMode.face, configuration.polygonMode.mode);
    // TODO: For test
    // glDrawArrays(GL_TRIANGLES, 0, 36);
    glDrawElements(GL_TRIANGLES, meshNode->vertexArrayIndices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}
}