#pragma once 

#include <vector>

#include <scene/Component.hpp>
#include <opengl/RenderPipeline.hpp>

namespace gl {
 
    template <typename Attributes = attributes::Cases>
    using RenderComponent = scene::ContainerComponent<std::vector<RenderPipeline<Attributes>>>;
}