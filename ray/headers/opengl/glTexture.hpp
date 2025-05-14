#pragma once

#include <functional>

#include "GL/glew.h"

#include "Tex.hpp"
#include "glCommon.hpp"

namespace gl {
    using TexDeleter = std::function<void (ID *)>;

    struct TextureMode {
        GLenum wrapModeS = GL_REPEAT;
        GLenum wrapModeT = GL_REPEAT;
        GLenum magnifyingFilter = GL_LINEAR;
        GLenum minifyingFilter = GL_LINEAR;
        float border[4] = {1.f, 1.f, 1.f, 1.f};
        bool mipmaps = true;

        GLenum bitFormat = GL_RGB; 
    };

    struct Texture final {
        TextureMode mode = TextureMode();
        std::unique_ptr<TexData> texData;
        ID id = 0;

        Texture(
            TextureMode mode,
            std::unique_ptr<TexData> texData
        ) : mode(mode), texData(std::move(texData)) {}

        Texture(
            Texture &&other
        ) : mode(other.mode), texData(std::move(other.texData)), id(other.id) {
            other.id = 0;
        }

        Texture& operator=(Texture &&other) {
            this->id = other.id;
            this->mode = other.mode;
            this->texData = std::move(other.texData);
            other.id = 0;
            return *this;
        }

        ~Texture() {
            if (id) glDeleteTextures(1, &id);
        }
    };
}