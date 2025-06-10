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
#include "glCommon.hpp"
#include "opengl/glTexture.hpp"
#include "scene/Mesh.hpp"

namespace gl {

using EBO = std::vector<unsigned int>;

struct Configuration {
    struct PolygonMode {
        GLenum face = GL_FRONT;
        GLenum mode = GL_FILL;
    };

    PolygonMode polygonMode;
};

void bindTextures(std::vector<gl::Texture> const &textures);
void activateMaterial(Material const &material);

template <typename Attribute>
struct RenderObject {
    RenderObject(
        Configuration configuration,
        std::shared_ptr<scene::Mesh<Attribute>> meshNode,
        Material& material,
        bool debug = false
    );

    RenderObject(RenderObject<Attribute> const &) = delete;
    RenderObject<Attribute>& operator=(RenderObject<Attribute> const&) = delete;

    RenderObject(RenderObject<Attribute> &&);
    RenderObject<Attribute>& operator=(RenderObject<Attribute>&& other);

    void setDebug(bool) noexcept;
    void setUniform(std::string const &key, attributes::UniformCases const &attr) noexcept;
    void prepare();
    void render() noexcept;

    ~RenderObject();

    ID vbo = 0;
    ID vao = 0;
    ID ebo = 0;
    ID tex;

    std::shared_ptr<scene::Mesh<Attribute>> meshNode;
    Material& material;

    Configuration configuration;

    bool debug;
};

template<typename Attribute>
RenderObject<Attribute>::RenderObject(
    Configuration configuration,
    std::shared_ptr<scene::Mesh<Attribute>> meshNode,
    Material& material,
    bool debug
) 
    : configuration(configuration)
    , meshNode(meshNode)
    , material(material) {
    setDebug(debug);
}

template<typename Attribute>
RenderObject<Attribute>::RenderObject(RenderObject<Attribute> &&other)
    : vao(other.vao)
    , ebo(other.ebo)
    , vbo(other.vbo)
    , configuration(other.configuration)
    , meshNode(std::move(other.meshNode))
    , material(other.material)
    , debug(other.debug)
{
    other.vbo = 0;
    other.ebo = 0;
    other.vao = 0;
}

template<typename Attribute>
RenderObject<Attribute>& RenderObject<Attribute>::operator=(RenderObject<Attribute>&& other) {
    this->configuration = configuration;
    this->meshNode = std::move(other.meshNode);
    this->material = other.material;
    this->debug = debug;
    this->vao = other.vao;
    this->vbo = other.vbo;
    this->ebo = other.ebo;
    other.vao = 0;
    other.ebo = 0;
    other.vbo = 0;
}

template<typename Attribute>
RenderObject<Attribute>::~RenderObject() {
    if (vbo) glDeleteBuffers(1, &vbo);
    if (ebo) glDeleteBuffers(1, &ebo);
    if (vao) glDeleteVertexArrays(1, &vao);
}

template<typename Attribute>
void RenderObject<Attribute>::prepare() {
    bindTextures(material.ambient);
    bindTextures(material.diffuse);
    bindTextures(material.specular);
    bindTextures(material.normals);

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
    bindAttributes<Attribute>();

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
void RenderObject<Attribute>::setDebug(bool debug) noexcept {
    this->debug = debug;
}

template<typename Attribute>
void RenderObject<Attribute>::render() noexcept {
    activateMaterial(material);
    
    glBindVertexArray(vao);
    glPolygonMode(configuration.polygonMode.face, configuration.polygonMode.mode);
    // TODO: For test
    // glDrawArrays(GL_TRIANGLES, 0, 36);
    glDrawElements(GL_TRIANGLES, meshNode->vertexArrayIndices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}
}