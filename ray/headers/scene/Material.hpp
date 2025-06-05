#pragma once

#include "scene/Tex.hpp"

namespace scene {

using MaterialPtr = std::shared_ptr<MaterialPtr>;

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
}