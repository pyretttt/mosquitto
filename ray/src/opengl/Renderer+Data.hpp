#pragma once

#include "GL/glew.h"
#include "SDL_opengl.h"
#include "SDL.h"

namespace gl {
    inline unsigned int cubeMapTextureId;

    inline void glDataGenerate() {
        glGenTextures(1, &cubeMapTextureId);
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubeMapTextureId);
        
    }
}