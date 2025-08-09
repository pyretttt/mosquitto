#include <memory>
#include <filesystem>
#include <variant>

#include "Core.hpp"
#include "opengl/RenderScene.hpp"
#include "scene/Material.hpp"
#include "scene/Node.hpp"
#include "scene/Mesh.hpp"
#include "scene/Tex.hpp"
#include "scene/MeshComponent.hpp"
#include "scene/AttributesInfoComponent.hpp"
#include "opengl/Uniforms.hpp"
#include "opengl/glCommon.hpp"
#include "opengl/glRenderComponent.hpp"
#include "opengl/glTexture.hpp"
#include "opengl/glFramebuffers.hpp"
#include "opengl/Shader.hpp"
#include "opengl/Shading.hpp"

namespace {
    auto toGlMesh(
        std::shared_ptr<scene::Mesh<>> nativeMesh,
        std::unordered_map<scene::ID, gl::AttachmentCases> const &attachments
    ) -> decltype(auto) {
        gl::AttachmentCases attachment = std::visit(overload {
            [&](scene::MaterialAttachment const &attachment) {
                return attachments.at(attachment.id);
            },
            [&](std::monostate _) {
                return gl::AttachmentCases(std::monostate());
            }
        }, nativeMesh->attachment);

        return std::make_shared<scene::Mesh<attributes::Cases, gl::AttachmentCases>>(
            nativeMesh->vertexArray,
            nativeMesh->vertexArrayIndices,
            attachment,
            InstanceIdGenerator<scene::Mesh<attributes::Cases, gl::AttachmentCases>>::getInstanceId()
        );
    }

    auto attachRenderPipelines(
        scene::ScenePtr scene,
        gl::PipelineConfiguration configuration,
        std::unordered_map<scene::ID, gl::AttachmentCases> const &attachments,
        gl::Shading shading
    ) -> decltype(auto) {
        for (auto const &[id, node] : scene->nodes) {
            std::vector<gl::RenderPipeline<>> renderPipelines;

            std::shared_ptr<scene::MeshComponent<>> meshComponent = node->getComponent<scene::MeshComponent<>>();
            if (meshComponent == nullptr) {
                assert("No mesh associated");
                continue;
            }
            std::vector<scene::MeshPtr<>> &meshes = meshComponent->value;
            for (auto &mesh : meshes) {
                auto renderPipeline = gl::RenderPipeline<>(
                    configuration,
                    toGlMesh(mesh, attachments),
                    gl::shaders.at(shading)
                );
                renderPipelines.emplace_back(std::move(renderPipeline));
            }
            node->addComponent<gl::RenderComponent<>>(
                InstanceIdGenerator<gl::RenderComponent<>>::getInstanceId(),
                std::move(renderPipelines)
            );
        }
    }

    auto toGlTexture(scene::TexturePtr texture, size_t unitIndex) -> decltype(auto) {
        return std::make_shared<gl::Texture>(texture, unitIndex);
    }

    auto makeGlAttachments(scene::ScenePtr scene) -> decltype(auto) {
        std::unordered_map<scene::ID, gl::AttachmentCases> glAttachments;
        for (auto const &[id, attachment] : scene->attachments) {
            gl::AttachmentCases glAttachment = std::visit(overload {
                [&](scene::MaterialAttachment const &materialAttachment) {
                    auto material = materialAttachment.material;
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
                    return gl::AttachmentCases(gl::MaterialAttachment(
                        std::make_shared<gl::Material>(
                            id,
                            material->ambientColor,
                            material->shiness,
                            std::move(ambient),
                            std::move(specular),
                            std::move(diffuse),
                            std::move(normals)
                        ),
                        id
                    ));
                },
                [](std::monostate) { return gl::AttachmentCases(std::monostate()); }
            }, attachment);
            glAttachments.emplace(
                std::make_pair(id, glAttachment)
            );
        }

        return glAttachments;
    }
}

gl::RenderScene::RenderScene(
    scene::ScenePtr scene,
    gl::PipelineConfiguration configuration,
    gl::FramebufferInfo framebufferInfo,
    gl::Shading shading
)
    : scene(scene)
    , configuration(configuration)
    , framebufferInfo(framebufferInfo)
    , shading(shading) {}


void gl::RenderScene::prepare() {
    auto attachments = makeGlAttachments(scene);
    attachRenderPipelines(scene, configuration, attachments, shading);

    for (auto const &[nodeId, node] : scene->nodes) {
        auto const &renderComponent = node->getComponent<gl::RenderComponent<>>();
        if (renderComponent) {
            for (auto &renderPipeline : renderComponent->value) {
                renderPipeline.prepare();
            }
        } else {
            std::cout << "Skipping render preparation of node with (ID) " << nodeId << std::endl;
        }
    }
}

void gl::RenderScene::render() const {
    glBindFramebuffer(GL_FRAMEBUFFER, framebufferInfo.fbo());

    if (actions.preRender) 
        actions.preRender();

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

    for (auto const &[nodeId, node] : scene->nodes) {
        ShaderPtr shader;
        auto component = node->getComponent<scene::AttributesInfoComponent>();
        auto const &renderComponent = node->getComponent<gl::RenderComponent<>>();
        if (renderComponent) {
            for (auto const &renderPipeline : renderComponent->value) {
                if (component && renderPipeline.shader != shader) {
                    shader = renderPipeline.shader;
                    for (auto const &[key, attribute] : component->value) {
                        shader->setUniform(key, attributes::Cases(attribute));
                    }
                }

                renderPipeline.render();
            }
        }
    }

    if (actions.postRender) 
        actions.postRender();
}