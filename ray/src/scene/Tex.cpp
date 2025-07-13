#include "scene/Tex.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "3rd_party/stb_image.h"

scene::TexData scene::loadTextureData(
    std::filesystem::path path,
    bool flipVertically
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
    TexDataPointer texPtr = TexDataPointer(
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