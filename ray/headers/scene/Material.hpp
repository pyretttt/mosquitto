#pragma once

#include "scene/Tex.hpp"

namespace scene {

struct Material {
    Material(
        std::vector<TexturePtr> ambient,
        std::vector<TexturePtr> diffuse,
        std::vector<TexturePtr> specular,
        std::vector<TexturePtr> normals
    );

    std::vector<TexturePtr> ambient;
    std::vector<TexturePtr> diffuse;
    std::vector<TexturePtr> specular;
    std::vector<TexturePtr> normals;
};

using MaterialPtr = std::shared_ptr<Material>;
}