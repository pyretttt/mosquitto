#pragma once

#include <utility>

struct RendererInitConfig {
    std::pair<int, int> resolution;
};

class Renderer {
public:
    virtual void update() const = 0;
    virtual void render() const = 0;
    virtual ~Renderer() = 0;
};
