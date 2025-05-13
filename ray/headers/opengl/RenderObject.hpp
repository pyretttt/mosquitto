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
        std::vector<Attribute> vertexArray,
        EBO vertexArrayIndices,
        Configuration configuration,
        bool eagerInit,
        bool debug = false,
        std::shared_ptr<std::vector<Texture>> textures = nullptr
    );

    RenderObject(RenderObject<Attribute> const &) = delete;
    RenderObject<Attribute>& operator=(RenderObject<Attribute> const&) = delete;

    RenderObject(RenderObject<Attribute> &&);
    RenderObject<Attribute>& operator=(RenderObject<Attribute>&& other);

    void setDebug(bool) noexcept;

    void setUniform(std::string const &key, attributes::BaseCases const &attr) noexcept;

    void prepare();
    void render() noexcept;

    ~RenderObject();

    ID vbo = 0;
    ID vao = 0;
    ID ebo = 0;
    ID tex;

    std::vector<Attribute> vertexArray;
    EBO vertexArrayIndices;
    std::shared_ptr<std::vector<Texture>> textures;

    Configuration configuration;

    bool debug;
};

template<typename Attribute>
RenderObject<Attribute>::RenderObject(
    std::vector<Attribute> vertexArray,
    EBO vertexArrayIndices,
    Configuration configuration,
    bool eagerInit,
    bool debug,
    std::shared_ptr<std::vector<Texture>> textures
) 
    : configuration(configuration)
    , vertexArray(std::move(vertexArray))
    , vertexArrayIndices(std::move(vertexArrayIndices))
    , textures(textures) {
    setDebug(debug);
    if (eagerInit) {
        prepare();
    }
}

template<typename Attribute>
RenderObject<Attribute>::RenderObject(RenderObject<Attribute> &&other)
    : vertexArray(std::move(vertexArray))
    , vertexArrayIndices(std::move(vertexArrayIndices))
    , vao(other.vao)
    , ebo(other.ebo)
    , vbo(other.vbo)
    , debug(other.debug)
    , configuration(other.configuration)
    , textures(std::move(other.textures))
{
    other.vbo = 0;
    other.ebo = 0;
    other.vao = 0;
}

template<typename Attribute>
RenderObject<Attribute>& RenderObject<Attribute>::operator=(RenderObject<Attribute>&& other) {
    this->vertexArray = std::move(other.vertexArray);
    this->vertexArrayIndices = std::move(other.vertexArrayIndices);
    this->vao = other.vao;
    this->vbo = other.vbo;
    this->ebo = other.ebo;
    this->debug = debug;
    this->configuration = configuration;
    this->textures = std::move(other.textures);
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

    std::for_each(
        textures->begin(),
        textures->end(),
        [](auto &texture) {
            glBindTexture(GL_TEXTURE_2D, texture.id);
            glTexImage2D(
                GL_TEXTURE_2D, 
                0, 
                GL_RGB, 
                texture.texData->width,
                texture.texData->height,
                0,
                GL_RGB,
                GL_UNSIGNED_BYTE,
                texture.texData->ptr.get()
            );
            if (texture.mode.mipmaps) {
                glGenerateMipmap(GL_TEXTURE_2D);
            }
            glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, texture.mode.border);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, texture.mode.wrapModeS);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, texture.mode.wrapModeT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, texture.mode.minifyingFilter);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, texture.mode.magnifyingFilter);
        }
    );


    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(
        GL_ARRAY_BUFFER, 
        vertexArray.size() * sizeof(Attribute),
        vertexArray.data(), 
        GL_STATIC_DRAW
    );
    bindAttributes<Attribute>();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER, 
        sizeof(EBO::value_type) * vertexArrayIndices.size(),
        vertexArrayIndices.data(),
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
    glBindVertexArray(vao);
    
    if (!textures->empty()) {
        glBindTexture(GL_TEXTURE_2D, textures->front().id);
    }

    glPolygonMode(configuration.polygonMode.face, configuration.polygonMode.mode);
    glDrawElements(GL_TRIANGLES, vertexArrayIndices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}
}