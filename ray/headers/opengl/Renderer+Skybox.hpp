#pragma once

#include "GL/glew.h"
#include "SDL_opengl.h"
#include "SDL.h"

#include "Core.hpp"
#include "Lazy.hpp"
#include "scene/MeshComponent.hpp"
#include "scene/Scene.hpp"
#include "scene/Node.hpp"
#include "scene/Mesh.hpp"
#include "scene/Tex.hpp"
#include "opengl/glFramebuffers.hpp"
#include "opengl/RenderScene.hpp"

namespace gl {
    inline unsigned int cubeMapTextureId;

    inline void glDataGenerate() {
        glGenTextures(1, &cubeMapTextureId);
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubeMapTextureId);
        
        auto skyboxFiles = std::vector<std::string>({
            "right.jpg",
            "left.jpg",
            "top.jpg",
            "bottom.jpg",
            "front.jpg",
            "back.jpg"
        });

        std::filesystem::path path("resources");
        for (size_t i = 0; i < skyboxFiles.size(); i++) {
            auto texture = scene::loadTextureData(path / "textures" / "skybox" / skyboxFiles[i], false);
            glTexImage2D(
                GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 
                0, 
                GL_RGB, 
                texture.width, 
                texture.height, 
                0, 
                GL_RGB, 
                GL_UNSIGNED_BYTE,
                texture.ptr.get()
            );
        }

        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
    }
    
    auto cubeMesh = std::make_shared<scene::Mesh<attributes::Cases>>(
        std::vector<attributes::Cases>({
            attributes::Vec3 {-1.0f,  1.0f, -1.0f},
            attributes::Vec3 {-1.0f, -1.0f, -1.0f},
            attributes::Vec3 {1.0f, -1.0f, -1.0f},
            attributes::Vec3 {1.0f, -1.0f, -1.0f},
            attributes::Vec3 {1.0f,  1.0f, -1.0f},
            attributes::Vec3 {-1.0f,  1.0f, -1.0f},
            
            attributes::Vec3 {-1.0f, -1.0f,  1.0f},
            attributes::Vec3 {-1.0f, -1.0f, -1.0f},
            attributes::Vec3 {-1.0f,  1.0f, -1.0f},
            attributes::Vec3 {-1.0f,  1.0f, -1.0f},
            attributes::Vec3 {-1.0f,  1.0f,  1.0f},
            attributes::Vec3 {-1.0f, -1.0f,  1.0f},

            attributes::Vec3 {1.0f, -1.0f, -1.0f},
            attributes::Vec3 {1.0f, -1.0f,  1.0f},
            attributes::Vec3 {1.0f,  1.0f,  1.0f},
            attributes::Vec3 {1.0f,  1.0f,  1.0f},
            attributes::Vec3 {1.0f,  1.0f, -1.0f},
            attributes::Vec3 {1.0f, -1.0f, -1.0f},

            attributes::Vec3 {-1.0f, -1.0f,  1.0f},
            attributes::Vec3 {-1.0f,  1.0f,  1.0f},
            attributes::Vec3 {1.0f,  1.0f,  1.0f},
            attributes::Vec3 {1.0f,  1.0f,  1.0f},
            attributes::Vec3 {1.0f, -1.0f,  1.0f},
            attributes::Vec3 {-1.0f, -1.0f,  1.0f},

            attributes::Vec3 {-1.0f,  1.0f, -1.0f},
            attributes::Vec3 {1.0f,  1.0f, -1.0f},
            attributes::Vec3 {1.0f,  1.0f,  1.0f},
            attributes::Vec3 {1.0f,  1.0f,  1.0f},
            attributes::Vec3 {-1.0f,  1.0f,  1.0f},
            attributes::Vec3 {-1.0f,  1.0f, -1.0f},

            attributes::Vec3 {-1.0f, -1.0f, -1.0f},
            attributes::Vec3 {-1.0f, -1.0f,  1.0f},
            attributes::Vec3 {1.0f, -1.0f, -1.0f},
            attributes::Vec3 {1.0f, -1.0f, -1.0f},
            attributes::Vec3 {-1.0f, -1.0f,  1.0f},
            attributes::Vec3 {1.0f, -1.0f, 1.0f}
        }),
        std::vector<unsigned int>(),
        std::monostate(),
        InstanceIdGenerator<scene::Mesh<attributes::Cases>>::getInstanceId()
    );

    scene::NodePtr cubeNode = modified(std::make_shared<scene::Node>(
        InstanceIdGenerator<scene::Node>::getInstanceId()
    ), [](std::shared_ptr<scene::Node> const &node) {
        node->addComponent<scene::MeshComponent<>>(
            InstanceIdGenerator<scene::MeshComponent<>>::getInstanceId(),
            std::vector<decltype(cubeMesh)>({cubeMesh})
        );
    });

    scene::ScenePtr cubeScene = std::make_shared<scene::Scene>(
        std::unordered_map<scene::ID, scene::NodePtr>({
            std::make_pair(cubeNode->identifier, cubeNode)
        }),
        std::unordered_map<scene::ID, scene::MaterialPtr>({}),
        std::unordered_map<scene::TexturePath, scene::TexturePtr>({}),
        std::unordered_map<scene::ID, scene::AttachmentCases>({})
    );

    gl::RenderScene makeCubeRenderScene() {
        glDataGenerate();
        return gl::RenderScene(
            cubeScene,
            gl::PipelineConfiguration {
                .polygonMode = gl::PipelineConfiguration::PolygonMode {
                    .face = GL_FRONT_FACE,
                    .mode = GL_FILL
                }
            },
            gl::FramebufferInfo {
                .framebuffer = gl::defaultFrameBuffer(),
                .useDepth = false,
                .useStencil = false
            },
            gl::Shading::Cube
        );
    }
    
    Lazy<gl::RenderScene> cubeRenderScene = Lazy<gl::RenderScene>(std::function<gl::RenderScene ()>(makeCubeRenderScene));
}