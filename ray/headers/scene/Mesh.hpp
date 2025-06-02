#pragma once

#include <memory>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "MathUtils.hpp"

namespace scene {

struct TextureIdentifier {
    size_t id;
};

template <typename Attr, typename TextureId = TextureIdentifier>
struct Mesh {
    Mesh(
        std::vector<Attr> vertexArray,
        std::vector<unsigned int> vertexArrayIndices,
        std::vector<TextureId> textures,
        size_t identifier
    );


    std::vector<Attr> vertexArray;
    std::vector<unsigned int> vertexArrayIndices;
    std::vector<TextureId> textures;
    size_t const identifier;
};

template <typename Attr, typename TextureId>
Mesh<Attr, TextureId>::Mesh(
    std::vector<Attr> vertexArray,
    std::vector<unsigned int> vertexArrayIndices,
    std::vector<TextureId> textures,
    size_t identifier
) 
    : vertexArray(std::move(vertexArray))
    , vertexArrayIndices(std::move(vertexArrayIndices))
    , textures(std::move(textures))
    , identifier(identifier) {
    if (this->vertexArrayIndices.empty()) {
        this->vertexArrayIndices = std::vector<unsigned int>(this->vertexArray.size());
        std::iota(this->vertexArrayIndices.begin(), this->vertexArrayIndices.end(), this->vertexArray.size());
    }
}
}