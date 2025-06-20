#pragma once

#include <functional>

#include "GL/glew.h"

#include "scene/Tex.hpp"
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
        scene::TexturePtr texturePtr;
        ID id = 0;

        Texture(
            scene::TexturePtr texturePtr
        ) : texturePtr(texturePtr) {}

        Texture(
            TextureMode mode,
            scene::TexturePtr texturePtr
        ) : mode(mode), texturePtr(texturePtr) {}

        Texture(
            Texture &&other
        ) : mode(other.mode), texturePtr(std::move(other.texturePtr)), id(other.id) {
            other.id = 0;
        }

        Texture& operator=(Texture &&other) {
            this->id = other.id;
            this->mode = other.mode;
            this->texturePtr = std::move(other.texturePtr);
            other.id = 0;
            return *this;
        }

        ~Texture() {
            if (id) glDeleteTextures(1, &id);
        }
    };

    using TexturePtr = std::shared_ptr<Texture>;
    
    struct Material {
        float shiness = 0;
        std::vector<TexturePtr> ambient;
        std::vector<TexturePtr> diffuse;
        std::vector<TexturePtr> specular;
        std::vector<TexturePtr> normals;
    };
}