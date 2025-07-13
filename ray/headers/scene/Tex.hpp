#pragma once

#include <memory>
#include <string>
#include <functional>
#include <filesystem>

#include "scene/Identifiers.hpp"

namespace scene {

using TexDataPointer = std::unique_ptr<
    unsigned char,
    std::function<void (unsigned char *)>
>;

struct TexData {
    int width;
    int height;
    int channels;
    TexDataPointer ptr;
};

TexData loadTextureData(
    std::filesystem::path path,
    bool flipVertically = true
);
}