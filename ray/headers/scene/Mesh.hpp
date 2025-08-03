#pragma once

#include <memory>
#include <numeric>
#include <unordered_map>
#include <vector>
#include <variant>

#include "MathUtils.hpp"
#include "scene/Material.hpp"
#include "scene/Identifiers.hpp"
#include "scene/Attachment.hpp"
#include "Attributes.hpp"

namespace scene {

template <typename Attr = attributes::Cases, typename Attachment = AttachmentCases>
struct Mesh {
    Mesh(
        std::vector<Attr> vertexArray,
        std::vector<unsigned int> vertexArrayIndices,
        Attachment attachment,
        ID identifier
    );

    std::vector<Attr> vertexArray;
    std::vector<unsigned int> vertexArrayIndices;
    Attachment attachment;
    ID const identifier;
};

template <typename Attr, typename Attachment>
Mesh<Attr, Attachment>::Mesh(
    std::vector<Attr> vertexArray,
    std::vector<unsigned int> vertexArrayIndices,
    Attachment attachment,
    ID identifier
) 
    : vertexArray(std::move(vertexArray))
    , vertexArrayIndices(std::move(vertexArrayIndices))
    , attachment(std::move(attachment))
    , identifier(identifier) {
    if (this->vertexArrayIndices.empty()) {
        this->vertexArrayIndices = std::vector<unsigned int>(this->vertexArray.size());
        std::iota(this->vertexArrayIndices.begin(), this->vertexArrayIndices.end(), this->vertexArray.size());
    }
}
}