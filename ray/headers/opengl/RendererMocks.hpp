#pragma once

#include "GL/glew.h"
#include "SDL_opengl.h"
#include "SDL.h"

#include "Attributes.hpp"
#include "Core.hpp"
#include "opengl/Renderer.hpp"
#include "opengl/RenderScene.hpp"
#include "scene/Mesh.hpp"
#include "sdlUtils.hpp"
#include "LoadTextFile.hpp"
#include "opengl/RenderObject.hpp"
#include "opengl/Shader.hpp"
#include "scene/Tex.hpp"
#include "MathUtils.hpp"
#include "scene/Node.hpp"
#include "scene/ShaderInfoComponent.hpp"
#include "opengl/Uniforms.hpp"
#include "Light.hpp"

namespace {
       scene::MaterialMeshPtr mesh = std::make_shared<scene::Mesh<attributes::AssimpVertex>>(
        std::vector<attributes::AssimpVertex>({
            // 1
            attributes::AssimpVertex {
                attributes::Vec3 {-0.5f, -0.5f, -0.5f}, 
                attributes::Vec3 {0.0f, 0.0f, -1.0f}, 
                attributes::Vec2 {0.0f, 0.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {0.5f, -0.5f, -0.5f,}, 
                attributes::Vec3 {0.0f, 0.0f, -1.0f}, 
                attributes::Vec2 {1.0f, 0.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {0.5f, 0.5f, -0.5f}, 
                attributes::Vec3 {0.0f, 0.0f, -1.0f}, 
                attributes::Vec2 {1.0f, 1.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {0.5f, 0.5f, -0.5f}, 
                attributes::Vec3 {0.0f, 0.0f, -1.0f}, 
                attributes::Vec2 {1.0f, 1.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {-0.5f, 0.5f, -0.5f}, 
                attributes::Vec3 {0.0f, 0.0f, -1.0f}, 
                attributes::Vec2 {0.0f, 1.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {-0.5f, -0.5f, -0.5f}, 
                attributes::Vec3 {0.0f, 0.0f, -1.0f}, 
                attributes::Vec2 {0.0f, 0.0f}
            },

            // 2
            attributes::AssimpVertex {
                attributes::Vec3 {-0.5f, -0.5f, 0.5f}, 
                attributes::Vec3 {0.0f, 0.0f, 1.0f}, 
                attributes::Vec2 {0.0f, 0.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {0.5f, -0.5f, 0.5f}, 
                attributes::Vec3 {0.0f, 0.0f, 1.0f}, 
                attributes::Vec2 {1.0f, 0.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {0.5f, 0.5f, 0.5f}, 
                attributes::Vec3 {0.0f, 0.0f, 1.0f}, 
                attributes::Vec2 {1.0f, 1.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {0.5f, 0.5f, 0.5f}, 
                attributes::Vec3 {0.0f, 0.0f, 1.0f}, 
                attributes::Vec2 {1.0f, 1.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {-0.5f, 0.5f, 0.5f}, 
                attributes::Vec3 {0.0f, 0.0f, 1.0f}, 
                attributes::Vec2 {0.0f, 1.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {-0.5f, -0.5f, 0.5f}, 
                attributes::Vec3 {0.0f, 0.0f, 1.0f}, 
                attributes::Vec2 {0.0f, 0.0f}
            },

            // 3
            attributes::AssimpVertex {
                attributes::Vec3 {-0.5f, 0.5f, 0.5f}, 
                attributes::Vec3 {-1.f, 0.f, 0.f}, 
                attributes::Vec2 {1.0f, 0.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {-0.5f, 0.5f, -0.5f}, 
                attributes::Vec3 {-1.f, 0.f, 0.f}, 
                attributes::Vec2 {1.0f, 1.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {-0.5f, -0.5f, -0.5f}, 
                attributes::Vec3 {-1.f, 0.f, 0.f}, 
                attributes::Vec2 {0.0f, 1.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {-0.5f, -0.5f, -0.5f}, 
                attributes::Vec3 {-1.f, 0.f, 0.f}, 
                attributes::Vec2 {0.0f, 1.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {-0.5f, -0.5f, 0.5f}, 
                attributes::Vec3 {-1.f, 0.f, 0.f}, 
                attributes::Vec2 {0.0f, 0.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {-0.5f, 0.5f, 0.5f}, 
                attributes::Vec3 {-1.f, 0.f, 0.f}, 
                attributes::Vec2 {1.0f, 0.0f}
            },

            // 4
            attributes::AssimpVertex {
                attributes::Vec3 {0.5f, 0.5f, 0.5f}, 
                attributes::Vec3 {1.f, 0.f, 0.f}, 
                attributes::Vec2 {1.0f, 0.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {0.5f, 0.5f, -0.5f}, 
                attributes::Vec3 {1.f, 0.f, 0.f}, 
                attributes::Vec2 {1.0f, 1.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {0.5f, -0.5f, -0.5f}, 
                attributes::Vec3 {1.f, 0.f, 0.f}, 
                attributes::Vec2 {0.0f, 1.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {0.5f, -0.5f, -0.5f}, 
                attributes::Vec3 {1.f, 0.f, 0.f}, 
                attributes::Vec2 {0.0f, 1.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {0.5f, -0.5f, 0.5f}, 
                attributes::Vec3 {1.f, 0.f, 0.f}, 
                attributes::Vec2 {0.0f, 0.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {0.5f, 0.5f, 0.5f}, 
                attributes::Vec3 {1.f, 0.f, 0.f}, 
                attributes::Vec2 {1.0f, 0.0f}
            },
            // 5
            attributes::AssimpVertex {
                attributes::Vec3 {-0.5f, -0.5f, -0.5f}, 
                attributes::Vec3 {0.f, -1.f, 0.f}, 
                attributes::Vec2 {0.0f, 1.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {0.5f, -0.5f, -0.5f}, 
                attributes::Vec3 {0.f, -1.f, 0.f}, 
                attributes::Vec2 {1.0f, 1.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {0.5f, -0.5f, 0.5f}, 
                attributes::Vec3 {0.f, -1.f, 0.f}, 
                attributes::Vec2 {1.0f, 0.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {0.5f, -0.5f, 0.5f}, 
                attributes::Vec3 {0.f, -1.f, 0.f}, 
                attributes::Vec2 {1.0f, 0.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {-0.5f, -0.5f, 0.5f}, 
                attributes::Vec3 {0.f, -1.f, 0.f}, 
                attributes::Vec2 {0.0f, 0.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {-0.5f, -0.5f, -0.5f}, 
                attributes::Vec3 {0.f, -1.f, 0.f}, 
                attributes::Vec2 {0.0f, 1.0f}
            },
            // 6
            attributes::AssimpVertex {
                attributes::Vec3 {-0.5f, 0.5f, -0.5f}, 
                attributes::Vec3 {0.f, 1.f, 0.f}, 
                attributes::Vec2 {0.0f, 1.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {0.5f, 0.5f, -0.5f}, 
                attributes::Vec3 {0.f, 1.f, 0.f}, 
                attributes::Vec2 {1.0f, 1.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {0.5f, 0.5f, 0.5f}, 
                attributes::Vec3 {0.f, 1.f, 0.f}, 
                attributes::Vec2 {1.0f, 0.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {0.5f, 0.5f, 0.5f}, 
                attributes::Vec3 {0.f, 1.f, 0.f}, 
                attributes::Vec2 {1.0f, 0.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {-0.5f, 0.5f, 0.5f}, 
                attributes::Vec3 {0.f, 1.f, 0.f}, 
                attributes::Vec2 {0.0f, 0.0f}
            },
            attributes::AssimpVertex {
                attributes::Vec3 {-0.5f, 0.5f, -0.5f}, 
                attributes::Vec3 {0.f, 1.f, 0.f}, 
                attributes::Vec2 {0.0f, 1.0f}
            },
            
        }),
        std::vector<unsigned int>({/*{
            0, 1, 3,
            1, 2, 3 
        }*/}),
        scene::MaterialIdentifier(),
        0
    );
 
    std::shared_ptr<scene::Node> node = std::make_shared<scene::Node>(
        InstanceIdGenerator<scene::Node>::getInstanceId(),
        std::vector<scene::MaterialMeshPtr>({mesh})
    );

    gl::Material material = gl::Material {
        .ambient = modified(std::vector<gl::TexturePtr>(), [](std::vector<gl::TexturePtr> &vec) {
            vec.emplace_back(
                std::make_shared<gl::Texture>(
                    gl::TextureMode(), 
                    std::make_unique<scene::TexData>(
                        scene::loadTextureData(
                            fs::path("resources").append("textures").append("container.jpg")
                        )
                    )
                )
            );
            vec.emplace_back(
                std::make_shared<gl::Texture>(
                    gl::TextureMode {.bitFormat = GL_RGBA}, 
                    std::make_unique<scene::TexData>(
                        scene::loadTextureData(
                            fs::path("resources").append("textures").append("awesomeface.png")
                        )
                    )
                )
            );
        }),
    };

    scene::MaterialPtr mockMaterial = std::make_shared<scene::Material>(
        attributes::Vec3({0.1, 0.2, 0.3}),
        0.f,
        std::vector({
            std::make_shared<scene::TexData>(
                scene::loadTextureData(
                    fs::path("resources").append("textures").append("container.jpg")
                )
            )
        }),
        std::vector({
            std::make_shared<scene::TexData>(
                scene::loadTextureData(
                    fs::path("resources").append("textures").append("awesomeface.png")
                )
            )
        }),
        std::vector<scene::TexturePtr>(),
        std::vector<scene::TexturePtr>()
    );

    scene::ScenePtr mockScene = std::make_shared<scene::Scene>(
        scene::Scene(
            { std::make_pair(node->identifier, node) },
            { std::make_pair(0, mockMaterial) },
            { }
        )
    );

    gl::RenderScene mockRenderScene = gl::RenderScene(
        mockScene,
        config,
        gl::materialShader
    );
}

namespace {
    gl::Material glMockMaterial = modified(gl::Material(), [](gl::Material &material) {
        material.diffuse.emplace_back<gl::TexturePtr>(
            std::make_shared<gl::Texture>(
                gl::TextureMode(), 
                std::make_shared<scene::TexData>(
                    scene::loadTextureData(
                        fs::path("resources").append("textures").append("container.jpg")
                    )
                )
            )
        );
        material.specular.emplace_back<gl::TexturePtr>(
            std::make_shared<gl::Texture>(
                gl::TextureMode {.bitFormat = GL_RGBA}, 
                std::make_shared<scene::TexData>(
                    scene::loadTextureData(
                        fs::path("resources").append("textures").append("awesomeface.png")
                    )
                )
            )
        );
    });

    gl::RenderObject mockRenderObject = gl::RenderObject<attributes::AssimpVertex>(
        gl::Configuration(),
        mesh,
        glMockMaterial
    );
}