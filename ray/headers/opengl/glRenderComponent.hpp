#pragma once 

#include <scene/Component.hpp>
#include <opengl/RenderPipeline.hpp>

namespace gl {
 
    template <typename Attributes = attributes::Cases>
    using RenderComponent = scene::ContainerComponent<RenderPipeline<Attributes>>;
}