#include <memory>
#include <filesystem>
#include <variant>

#include "Core.hpp"
#include "opengl/RenderScene.hpp"
#include "scene/Material.hpp"
#include "scene/Node.hpp"
#include "scene/Mesh.hpp"
#include "scene/AttributesInfoComponent.hpp"
#include "opengl/Uniforms.hpp"
#include "opengl/glCommon.hpp"

namespace {
    auto toGlMesh(
        std::shared_ptr<scene::Mesh<>> nativeMesh,
        std::unordered_map<scene::MaterialId, gl::Material> const &materials
    ) -> decltype(auto) {
        gl::AttachmentCases attachment = std::visit(overload {
            [&](scene::MaterialAttachment &attachment) {
                return gl::AttachmentCases(gl::MaterialAttachment(materials.at(attachment.id), attachment.id));
            },
            [&](std::monostate) {
                return std::monostate();
            }
        }, nativeMesh->attachment);

        return std::make_shared<scene::Mesh<attributes::Cases, gl::AttachmentCases>>(
            nativeMesh->vertexArray,
            nativeMesh->vertexArrayIndices,
            attachment,
            InstanceIdGenerator<scene::Mesh<attributes::Cases, gl::AttachmentCases>>::getInstanceId()
        );
    }

    auto makeRenderPipelines(
        scene::ScenePtr scene,
        gl::PipelineConfiguration configuration,
        std::unordered_map<scene::MaterialId, gl::Material> const &materials
    ) -> decltype(auto) {
        std::vector<gl::RenderPipelineInfo> result;

        for (auto const &[id, node] : scene->nodes) {
            std::shared_ptr<scene::MeshComponent<>> meshComponent = node->getComponent<scene::MeshComponent<>>();
            if (meshComponent == nullptr) {
                assert("No mesh associated");
            }
            auto const &meshes = meshComponent->;
            for (auto &mesh : meshes) {
                auto glMesh = toGlMesh(mesh, materials);
                result.emplace_back(
                    static_cast<size_t>(id),
                    gl::RenderPipeline<>(
                        configuration,
                        glMesh
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
        std::unordered_map<scene::MaterialId, std::shared_ptr<gl::Material>> result;
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
                    std::make_shared<gl::Material>(
                        static_cast<scene::MaterialId>(id),
                        material->ambientColor,
                        material->shiness,
                        std::move(ambient),
                        std::move(specular),
                        std::move(diffuse),
                        std::move(normals)
                    )
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
    this->renderPipelines = makeRenderPipelines(scene, configuration, materials);
    shader->setup();

    std::for_each(renderPipelines.begin(), renderPipelines.end(), [](gl::RenderPipelineInfo &RenderPipelineInfo) {
        RenderPipelineInfo.renderPipeline.prepare();
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

    for (auto const &pipelineInfo : renderPipelines) {
        shader->use();
        auto const &node = scene->nodes.at(pipelineInfo.nodeId);
        if (auto component = node->getComponent<scene::AttributesInfoComponent>()) {
            auto shaderInfo = std::static_pointer_cast<scene::AttributesInfoComponent>(component.value());
            for (auto const &[key, attribute] : shaderInfo.value()) {
                shader->setUniform(key, attributes::Cases(attribute));
            }
        }

        auto const &material = pipelineInfo.renderPipeline;
        shader->setUniform("material", material);
        pipelineInfo.renderPipeline.render();
    }
}