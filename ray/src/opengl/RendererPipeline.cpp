#include "opengl/RenderPipeline.hpp"
#include "opengl/glCommon.hpp"

void gl::bindTextures(std::vector<gl::TexturePtr> const &textures) {
    std::for_each(
        textures.begin(), 
        textures.end(),
        [](auto &texture) {
            if (!texture->id)
                glGenTextures(1, &(texture->id));
        }
    );

    for (size_t i = 0; i < textures.size(); i++) {
        auto const &tex = textures.at(i);
        glBindTexture(GL_TEXTURE_2D, tex->id);
        glTexImage2D(
            GL_TEXTURE_2D, 
            0, 
            GL_RGB, 
            tex->texturePtr->width,
            tex->texturePtr->height,
            0,
            tex->mode.bitFormat,
            GL_UNSIGNED_BYTE,
            tex->texturePtr->ptr.get()
        );
        if (tex->mode.mipmaps) {
            glGenerateMipmap(GL_TEXTURE_2D);
        }
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, tex->mode.border);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, tex->mode.wrapModeS);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, tex->mode.wrapModeT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, tex->mode.minifyingFilter);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, tex->mode.magnifyingFilter);
    }
    glBindTexture(GL_TEXTURE_2D, 0);
}

void gl::activateMaterial(gl::Material const &material) {
    auto ambientsCounts = material.ambient.size();
    auto diffuseCounts = material.diffuse.size();
    auto specularCounts = material.specular.size();
    auto normalsCount = material.normals.size();

    for (size_t i = 0; i < ambientsCounts; i++) {
        glActiveTexture(GL_TEXTURE0 + material.ambient.at(i)->unitIndex);
        glBindTexture(GL_TEXTURE_2D, material.ambient.at(i)->id);
    }

    for (size_t i = 0; i < diffuseCounts; i++) {
        glActiveTexture(GL_TEXTURE0 + material.ambient.at(i)->unitIndex);
        glBindTexture(GL_TEXTURE_2D, material.diffuse.at(i)->id);
    }

    for (size_t i = 0; i < specularCounts; i++) {
        glActiveTexture(GL_TEXTURE0 + material.ambient.at(i)->unitIndex);
        glBindTexture(GL_TEXTURE_2D, material.specular.at(i)->id);
    }

    for (size_t i = 0; i < normalsCount; i++) {
        glActiveTexture(GL_TEXTURE0 + material.ambient.at(i)->unitIndex);
        glBindTexture(GL_TEXTURE_2D, material.normals.at(i)->id);
    }
    glActiveTexture(GL_TEXTURE0);
}