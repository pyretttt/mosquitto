#pragma once

#include <string>

namespace scene {


    using NodeId = size_t;
    using MaterialId = size_t;
    using MeshId = size_t;
    using TexturePath = std::string;
    using ComponentId = size_t;

    struct Material;
    using MaterialPtr = std::shared_ptr<Material>;

    struct MaterialIdentifier {
        MaterialId id = 0;
    };

    struct Node;
    using NodePtr = std::shared_ptr<Node>;

    struct TexData;
    using TexturePtr = std::shared_ptr<TexData>;

    struct Scene;
    using ScenePtr = std::shared_ptr<Scene>;

    struct Component;
    using ComponentPtr = std::unique_ptr<Component>;
}