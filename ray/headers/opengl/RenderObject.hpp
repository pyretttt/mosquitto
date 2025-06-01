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

template <typename Attribute>
struct RenderObject {
    RenderObject(
        Configuration configuration,
        std::shared_ptr<scene::Mesh<Attribute>> meshNode,
        std::shared_ptr<std::vector<Texture>> textures = nullptr,
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
    std::shared_ptr<std::vector<Texture>> textures;

    Configuration configuration;

    bool debug;
};

template<typename Attribute>
RenderObject<Attribute>::RenderObject(
    Configuration configuration,
    std::shared_ptr<scene::Mesh<Attribute>> meshNode,
    std::shared_ptr<std::vector<Texture>> textures,
    bool debug
) 
    : configuration(configuration)
    , meshNode(meshNode)
    , textures(textures) {
    setDebug(debug);
}

template<typename Attribute>
RenderObject<Attribute>::RenderObject(RenderObject<Attribute> &&other)
    : vao(other.vao)
    , ebo(other.ebo)
    , vbo(other.vbo)
    , configuration(other.configuration)
    , meshNode(std::move(other.meshNode))
    , textures(std::move(other.textures))
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
    this->textures = std::move(other.textures);
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
    std::for_each(
        textures->begin(), 
        textures->end(),
        [](auto &texture) {
            if (!texture.id)
                glGenTextures(1, &(texture.id));
        }
    );

    for (size_t i = 0; i < textures->size(); i++) {
        auto const &tex = textures->at(i);
        glBindTexture(GL_TEXTURE_2D, tex.id);
        glTexImage2D(
            GL_TEXTURE_2D, 
            0, 
            GL_RGB, 
            tex.texData->width,
            tex.texData->height,
            0,
            tex.mode.bitFormat,
            GL_UNSIGNED_BYTE,
            tex.texData->ptr.get()
        );
        if (tex.mode.mipmaps) {
            glGenerateMipmap(GL_TEXTURE_2D);
        }
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, tex.mode.border);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, tex.mode.wrapModeS);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, tex.mode.wrapModeT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, tex.mode.minifyingFilter);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, tex.mode.magnifyingFilter);
    }
    glBindTexture(GL_TEXTURE_2D, 0);

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
    for (size_t i = 0; i < textures->size(); i++) {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, textures->at(i).id);
    }
    
    glBindVertexArray(vao);
    glPolygonMode(configuration.polygonMode.face, configuration.polygonMode.mode);
    // TODO: For test
    // glDrawArrays(GL_TRIANGLES, 0, 36);
    glDrawElements(GL_TRIANGLES, meshNode->vertexArrayIndices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}
}