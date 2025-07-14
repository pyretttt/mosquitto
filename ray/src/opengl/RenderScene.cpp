#include <memory>
#include <filesystem>
#include <variant>

#include "opengl/RenderScene.hpp"
#include "scene/Material.hpp"
#include "scene/Node.hpp"
#include "scene/AttributesInfoComponent.hpp"
#include "opengl/Uniforms.hpp"
#include "opengl/glCommon.hpp"

namespace {
    auto makeRenderPipelines(
        scene::ScenePtr scene,
        gl::PipelineConfiguration configuration,
        std::unordered_map<scene::MaterialId, gl::Material> const &materials
    ) -> decltype(auto) {
        std::vector<gl::RenderPipelineInfo> result;

        for (auto const &[id, node] : scene->nodes) {
            for (auto &mesh : node->meshes) {
                if (!mesh->material.has_value()) {
                    continue;
                }
                auto materialId = mesh->material.value().id;
                if (!materials.contains(materialId)) {
                    throw "Material::NotFound";
                }
                result.emplace_back(
                    static_cast<size_t>(id),
                    gl::RenderPipeline<attributes::MaterialVertex>(
                        configuration,
                        mesh,
                        materials.at(materialId)
                    )
                );
            }
        }

        return result;
    }

    auto toGlTexture(scene::TexturePtr texture, size_t unitIndex) -> decltype(auto) {
        return std::make_shared<gl::Texture>(texture, unitIndex);
    }

    auto makeMaterials(scene::ScenePtr scene) -> decltype(auto) {
        std::unordered_map<scene::MaterialId, gl::Material> result;
        for (auto const &[id, material] : scene->materials) {
            std::vector<gl::TexturePtr> ambient, specular, diffuse, normals;
            auto unitIndex = gl::samplerLocationOffset;
            std::transform(
                material->ambient.begin(), 
                material->ambient.end(), 
                std::back_inserter(ambient),
                [&unitIndex](scene::TexturePtr &ptr) {
                    return toGlTexture(ptr, unitIndex++);
                }
            );
            std::transform(
                material->specular.begin(), 
                material->specular.end(), 
                std::back_inserter(specular),
                [&unitIndex](scene::TexturePtr &ptr) {
                    return toGlTexture(ptr, unitIndex++);
                }
            );
            std::transform(
                material->diffuse.begin(), 
                material->diffuse.end(), 
                std::back_inserter(diffuse),
                [&unitIndex](scene::TexturePtr &ptr) {
                    return toGlTexture(ptr, unitIndex++);
                }
            );
            std::transform(
                material->normals.begin(), 
                material->normals.end(), 
                std::back_inserter(normals),
                [&unitIndex](scene::TexturePtr &ptr) {
                    return toGlTexture(ptr, unitIndex++);
                }
            );

            result.emplace(
                std::make_pair(
                    static_cast<scene::MaterialId>(id), 
                    gl::Material {
                        .id = static_cast<scene::MaterialId>(id),
                        .ambientColor = material->ambientColor,
                        .shiness = material->shiness,
                        .ambient = std::move(ambient),
                        .specular = std::move(specular),
                        .diffuse = std::move(diffuse),
                        .normals = std::move(normals)
                    }
                )
            );
        }

        return result;
    }
}

gl::RenderScene::RenderScene(
    scene::ScenePtr scene,
    gl::PipelineConfiguration configuration,
    ShaderPtr shader,
    gl::FramebufferInfo framebufferInfo
)
    : scene(scene)
    , configuration(configuration)
    , shader(shader)
    , framebufferInfo(framebufferInfo) {}


void gl::RenderScene::prepare() {
    this->materials = makeMaterials(scene);
    this->pbrs = makeRenderPipelines(scene, configuration, materials);
    shader->setup();

    std::for_each(pbrs.begin(), pbrs.end(), [](gl::RenderPipelineInfo &RenderPipelineInfo) {
        RenderPipelineInfo.RenderPipeline.prepare();
    });
}

void gl::RenderScene::render() const {
    glBindFramebuffer(GL_FRAMEBUFFER, framebufferInfo.fbo());
    if (framebufferInfo.useStencil) {
        glEnable(GL_STENCIL_TEST);
    } else {
        glDisable(GL_STENCIL_TEST);
    }

    if (framebufferInfo.useDepth) {
        glEnable(GL_DEPTH_TEST);
    } else {
        glDisable(GL_DEPTH_TEST);
    }

    for (auto const &RenderPipelineInfo : pbrs) {
        shader->use();
        auto const &node = scene->nodes.at(RenderPipelineInfo.nodeId);
        if (auto component = node->getComponent<scene::AttributesInfoComponent>()) {
            auto shaderInfo = std::static_pointer_cast<scene::AttributesInfoComponent>(component.value());
            for (auto const &[key, attribute] : shaderInfo->uniforms) {
                shader->setUniform(key, attributes::Cases(attribute));
            }
        }

        auto const &material = RenderPipelineInfo.RenderPipeline.material;
        shader->setUniform("material", material);
        RenderPipelineInfo.RenderPipeline.render();
    }
}