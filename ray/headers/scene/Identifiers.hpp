#pragma once

#include <string>

namespace scene {
    using ID = size_t;
    using TexturePath = std::string;

    struct Material;
    using MaterialPtr = std::shared_ptr<Material>;

    struct Node;
    using NodePtr = std::shared_ptr<Node>;

    struct TexData;
    using TexturePtr = std::shared_ptr<TexData>;

    struct Scene;
    using ScenePtr = std::shared_ptr<Scene>;

    struct Component;
    using ComponentPtr = std::shared_ptr<Component>;
}