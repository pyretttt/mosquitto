#pragma once

#include <memory>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "MathUtils.hpp"
#include "scene/Material.hpp"
#include "scene/Identifiers.hpp"

namespace scene {

template <typename Attr, typename MaterialId = MaterialIdentifier>
struct Mesh {
    Mesh(
        std::vector<Attr> vertexArray,
        std::vector<unsigned int> vertexArrayIndices,
        std::optional<MaterialIdentifier> material,
        MeshId identifier
    );

    std::vector<Attr> vertexArray;
    std::vector<unsigned int> vertexArrayIndices;
    std::optional<MaterialIdentifier> material;
    MeshId const identifier;
};

template <typename Attr, typename TextureId>
Mesh<Attr, TextureId>::Mesh(
    std::vector<Attr> vertexArray,
    std::vector<unsigned int> vertexArrayIndices,
    std::optional<MaterialIdentifier> material,
    MeshId identifier
) 
    : vertexArray(std::move(vertexArray))
    , vertexArrayIndices(std::move(vertexArrayIndices))
    , material(std::move(material))
    , identifier(identifier) {
    if (this->vertexArrayIndices.empty()) {
        this->vertexArrayIndices = std::vector<unsigned int>(this->vertexArray.size());
        std::iota(this->vertexArrayIndices.begin(), this->vertexArrayIndices.end(), this->vertexArray.size());
    }
}
}