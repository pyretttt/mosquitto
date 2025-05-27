#pragma once

#include <memory>

namespace scene {

class Scene;

struct SceneNode {
    std::weak_ptr<Scene> scene;
    std::weak_ptr<SceneNode> parent;
};
}