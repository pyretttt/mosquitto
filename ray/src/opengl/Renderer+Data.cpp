#pragma once

#include "GL/glew.h"
#include "SDL_opengl.h"
#include "SDL.h"

#include "scene/Tex.hpp"

namespace gl {
    inline unsigned int cubeMapTextureId;

    inline void glDataGenerate() {
        glGenTextures(1, &cubeMapTextureId);
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubeMapTextureId);
        
        auto skyboxFiles = std::vector<std::string>({
            "right.jpg",
            "left.jpg",
            "top.jpg",
            "bottom.jpg",
            "back.jpg",
            "frong.jpg"
        });

        std::filesystem::path path("resources");
        for (size_t i = 0; i < skyboxFiles.size(); i++) {
            auto texture = scene::loadTextureData(path / "textures" / "skybox" / skyboxFiles[i]);
            glTexImage2D(
                GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 
                0, 
                GL_RGB, 
                texture.width, 
                texture.height, 
                0, 
                GL_RGB, 
                GL_UNSIGNED_BYTE,
                texture.ptr.get()
            );
        }

        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    }
}