#pragma once

#include <memory>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "MathUtils.hpp"
#include "Material.hpp"

namespace scene {

struct MaterialIdentifier {
    size_t id;
};

template <typename Attr, typename MaterialId = MaterialIdentifier>
struct Mesh {
    Mesh(
        std::vector<Attr> vertexArray,
        std::vector<unsigned int> vertexArrayIndices,
        std::optional<MaterialIdentifier> material,
        size_t identifier
    );

    std::vector<Attr> vertexArray;
    std::vector<unsigned int> vertexArrayIndices;
    std::optional<MaterialIdentifier> material;
    size_t const identifier;
};

template <typename Attr, typename TextureId>
Mesh<Attr, TextureId>::Mesh(
    std::vector<Attr> vertexArray,
    std::vector<unsigned int> vertexArrayIndices,
    std::optional<MaterialIdentifier> material,
    size_t identifier
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