#pragma once

#include <filesystem>

#include "LoadTextFile.hpp"

namespace gl {
struct RenderObject {
    RenderObject(
        std::filesystem::path vertexShader,
        std::filesystem::path fragmentShader
    );

    void prepare();
    void bind();
    void unbind();
    void render();
};
}