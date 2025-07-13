#pragma once

#include "scene/Component.hpp"

namespace scene {
    template<typename RenderPipeline>
    struct RenderComponent final : public Component {

       RenderComponent(
            RenderPipeline RenderPipeline,
            ComponentId id
       ) 
       : Component(id)
       , RenderPipeline(std::move(RenderPipeline)) {}

        void prepare();
        void render() const noexcept;

        RenderPipeline RenderPipeline;
    };

    template<typename RenderPipeline>
    void RenderComponent<RenderPipeline>::prepare() {
        RenderPipeline.prepare();
    }

    template<typename RenderPipeline>
    void RenderComponent<RenderPipeline>::render() const noexcept {
        RenderPipeline.render();
    }
}