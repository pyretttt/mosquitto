#pragma once

#include <memory>

namespace system {

class Scene;

struct SceneNode {


    std::weak_ptr<Scene> scene;
    std::weak_ptr<SceneNode> parent;

};
}