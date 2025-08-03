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

namespace {
    auto toGlMesh(
        std::shared_ptr<scene::Mesh<>> nativeMesh,
        std::unordered_map<scene::ID, std::shared_ptr<gl::Material>> const &materials
    ) -> decltype(auto) {
        gl::AttachmentCases attachment = std::visit(overload {
            [&](scene::MaterialAttachment const &attachment) {
                return gl::AttachmentCases(
                    gl::MaterialAttachment(
                        materials.at(attachment.id), 
                        attachment.id
                    )
                );
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
        std::unordered_map<scene::ID, std::shared_ptr<gl::Material>> const &materials
    ) -> decltype(auto) {
        for (auto const &[id, node] : scene->nodes) {
            std::vector<gl::RenderPipeline<>> renderPipelines;

            std::shared_ptr<scene::MeshComponent<>> meshComponent = node->getComponent<scene::MeshComponent<>>();
            if (meshComponent == nullptr) {
                assert("No mesh associated");
            }
            std::vector<scene::MeshPtr<>> &meshes = meshComponent->value;
            for (auto &mesh : meshes) {
                auto renderPipeline = gl::RenderPipeline<>(
                    configuration,
                    toGlMesh(mesh, materials),
                    gl::materialShader
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

    auto deriveAttachments(scene::ScenePtr scene) -> decltype(auto) {
        std::unordered_map<scene::ID, gl::AttachmentCases> glAttachments;
        for (auto const &[id, attachment] : scene->attachments) {
            std::visit(overload {
                [&](scene::MaterialAttachment const &materialAttachment){
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

                    glAttachments.emplace(
                        std::make_pair(
                            static_cast<scene::ID>(materialAttachment.id), 
                            std::make_shared<gl::Material>(
                                static_cast<scene::ID>(materialAttachment.id),
                                material->ambientColor,
                                material->shiness,
                                std::move(ambient),
                                std::move(specular),
                                std::move(diffuse),
                                std::move(normals)
                            )
                        )
                    );
                },
                [](std::monostate) {}
            },
            attachment);
        }

        return glAttachments;
    }
}

gl::RenderScene::RenderScene(
    scene::ScenePtr scene,
    gl::PipelineConfiguration configuration,
    gl::FramebufferInfo framebufferInfo
)
    : scene(scene)
    , configuration(configuration)
    , framebufferInfo(framebufferInfo) {}


void gl::RenderScene::prepare() {
    auto attachments = deriveAttachments(scene); // Stop point
    attachRenderPipelines(scene, configuration, materials);

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
}