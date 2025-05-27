#pragma once

#include <memory>
#include <string>
#include <functional>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#include "3rd_party/stb_image.h"

namespace scene {

using TexPointer = std::unique_ptr<
    unsigned char,
    std::function<void (unsigned char *)>
>;

struct TexData {
    int width;
    int height;
    int channels;
    TexPointer ptr;
};

TexData inline static loadTextureData(
    std::filesystem::path path,
    bool flipVertically = true
) {
    std::string pathStr = path.string();
    auto pathChars = pathStr.data();
    int width, height, channels;

    stbi_set_flip_vertically_on_load(flipVertically);
    unsigned char *data = stbi_load(
        pathChars,
        &width,
        &height,
        &channels,
        0
    );
    TexPointer texPtr = TexPointer(
        data,
        [](unsigned char *data) {
            stbi_image_free(data);
        }
    );
    return TexData {
        .width = width,
        .height = height,
        .channels = channels,
        .ptr = std::move(texPtr)
    };
}
}