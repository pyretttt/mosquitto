#pragma once

#include "scene/Component.hpp"

namespace scene {
    template<typename RenderObject>
    struct RenderComponent final : public Component {

       RenderComponent(
            RenderObject renderObject,
            ComponentId id
       ) 
       : Component(id)
       , renderObject(std::move(renderObject)) {}

        void prepare();
        void render() const noexcept;

        RenderObject renderObject;
    };

    template<typename RenderObject>
    void RenderComponent<RenderObject>::prepare() {
        renderObject.prepare();
    }

    template<typename RenderObject>
    void RenderComponent<RenderObject>::render() const noexcept {
        renderObject.render();
    }
}