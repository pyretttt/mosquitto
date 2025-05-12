#pragma once

#include <memory>
#include <string>
#include <functional>
#include <filesystem>

#include "3rd_party/stb_image.h"

using TexPointer = std::unique_ptr<
    unsigned char,
    std::function<void (unsigned char *)>
>;

struct Tex {
    int width;
    int height;
    int channels;
    TexPointer ptr;
};

Tex inline static loadText(
    std::filesystem::path path
) {
    std::string pathStr = path.string();
    auto pathChars = pathStr.data();
    int width, height, channels;
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
    return Tex {
        .width = width,
        .height = height,
        .channels = channels,
        .ptr = std::move(texPtr)
    };
}