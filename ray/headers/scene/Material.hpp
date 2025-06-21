#pragma once

#include "scene/Tex.hpp"
#include "scene/Identifiers.hpp"
#include "Attributes.hpp"

namespace scene {

struct Material {
    Material(
        attributes::Vec3 ambientColor,
        float shiness,
        std::vector<TexturePtr> ambient,
        std::vector<TexturePtr> diffuse,
        std::vector<TexturePtr> specular,
        std::vector<TexturePtr> normals
    );

    attributes::Vec3 ambientColor;
    float shiness = 0;
    std::vector<TexturePtr> ambient;
    std::vector<TexturePtr> diffuse;
    std::vector<TexturePtr> specular;
    std::vector<TexturePtr> normals;
};
}