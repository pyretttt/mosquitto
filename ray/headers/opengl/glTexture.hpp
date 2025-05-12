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
    };

    struct glTexture final {
        TextureMode mode;
        std::unique_ptr<Tex> texData;
        ID id = 0;

        ~glTexture() {
            if (id) glDeleteTextures(1, &id);
        }
    };
}