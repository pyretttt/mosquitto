#pragma once

#include "scene/Tex.hpp"
#include "scene/Identifiers.hpp"

namespace scene {

struct Material {
    Material(
        float shiness,
        std::vector<TexturePtr> ambient,
        std::vector<TexturePtr> diffuse,
        std::vector<TexturePtr> specular,
        std::vector<TexturePtr> normals
    );

    float shiness = 0;
    std::vector<TexturePtr> ambient;
    std::vector<TexturePtr> diffuse;
    std::vector<TexturePtr> specular;
    std::vector<TexturePtr> normals;
};
}