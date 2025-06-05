#pragma once

#include <memory>
#include <string>
#include <functional>
#include <filesystem>


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

using TexturePtr = std::shared_ptr<TexData>;

TexData loadTextureData(
    std::filesystem::path path,
    bool flipVertically = true
);
}