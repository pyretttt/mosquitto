#include "scene/Material.hpp"

scene::Material::Material(
    float shiness,
    std::vector<TexturePtr> ambient,
    std::vector<TexturePtr> diffuse,
    std::vector<TexturePtr> specular,
    std::vector<TexturePtr> normals
) 
    : ambient(std::move(ambient))
    , diffuse(std::move(diffuse))
    , specular(std::move(specular))
    , normals(std::move(normals)) {}