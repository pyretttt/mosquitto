#include <filesystem>
#include <array>
#include <vector>
#include <memory>

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

namespace fs = std::filesystem;

namespace {
    gl::Configuration config = gl::Configuration {
        .polygonMode = gl::Configuration::PolygonMode {
            .face = GL_FRONT_FACE,
            .mode = GL_FILL
        }
    };

    std::vector<LightSource> light = std::vector({
        LightSource {
            .position = ml::Vector3f({ 0.f, 0.f, 0.f }),
            .spotDirection = ml::Vector3f({ 0.f, 0.f, -1.f }),
            .ambient = ml::Vector3f({0.3f, 0.3f, 0.3f}),
            .diffuse = ml::Vector3f({0.35f, 0.35f, 0.35f}),
            .specular = ml::Vector3f({0.25f, 0.25f, 0.25f}),
            .cutoffRadians = ml::toRadians(20.f),
            .cutoffDecayRadians = ml::toRadians(12.5f),
            .attenuanceConstant = 1.f,
            .attenuanceLinear = 0.09f,
            .attenuanceQuadratic = 0.032f
        },
        LightSource {
            .position = ml::Vector3f({ -2.f, 1.f, -4.f }),
            .spotDirection = ml::Vector3f({ 2.f, -0.5f, -1.f }),
            .ambient = ml::Vector3f({0.3f, 0.3f, 0.3f}),
            .diffuse = ml::Vector3f({0.35f, 0.35f, 0.35f}),
            .specular = ml::Vector3f({0.25f, 0.25f, 0.25f}),
            .cutoffRadians = ml::toRadians(180.f),
            .cutoffDecayRadians = ml::toRadians(0.1f),
            .attenuanceConstant = 1.f,
            .attenuanceLinear = 0.09f,
            .attenuanceQuadratic = 0.032f
        }
    });

    void configureGl() noexcept {
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_STENCIL_TEST);

        glStencilMask(0xFF);
        glStencilFunc(GL_ALWAYS, 1, 0xFF);
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);

        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glFrontFace(GL_CCW);
    
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    }
}

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

namespace {
    scene::ScenePtr scenePtr = std::make_shared<scene::Scene>(
        scene::Scene::assimpImport(fs::path("resources").append("backpack").append("backpack.obj"))
    );

    gl::RenderScene renderScene = gl::RenderScene(
        scenePtr,
        config,
        gl::materialShader
    );
}

gl::Renderer::Renderer(
    std::shared_ptr<GlobalConfig> config,
    Lazy<Camera> camera
)
    : config(config)
    , camera(camera)
    , resolution(config->windowSize.value())
    , scene(scenePtr) {        
}

gl::Renderer::~Renderer() {
    SDL_GL_DeleteContext(glContext);
}

void gl::Renderer::prepareViewPort() {
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 32);

    if (!config->window) {
        auto window = SDL_CreateWindow(
            "SDL", 
            SDL_WINDOWPOS_UNDEFINED, 
            SDL_WINDOWPOS_UNDEFINED, 
            resolution.first, 
            resolution.second, 
            SDL_WINDOW_BORDERLESS | SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL
        );
        if (!window) {
            std::cerr << "Failed to create window" << std::endl;
            throw "GL::window_creation_failure";
        }
        config->window.reset(window);
    }
    glContext = SDL_GL_CreateContext(config->window.get());
    if (!glContext) {
        std::cerr << "Failed to create gl context" << std::endl;
        throw "GL::context_creation_failure";
    }

    glewExperimental = GL_TRUE;
    if (glewInit()) {
        throw "Glew_init_failure";
        
    };

    if (SDL_GL_SetSwapInterval(1) < 0) {
        std::cerr << "Failed to enable vsync" << std::endl;
        throw "GL::vsync_activation_failure";
    }

    SDL_GL_MakeCurrent(config->window.get(), glContext);
    glViewport(0, 0, resolution.first, resolution.second);

    // Switch for testing
    // shader->setup();
    // mockRenderObject.prepare();
    // mockRenderScene.prepare();
    renderScene.prepare();

    configureGl();
}

void gl::Renderer::processInput(Event event, float dt) {
    std::visit(overload {
        [&camera = this->camera, resolution = this->resolution, dt](SDL_Event event) {
            switch (event.type) {
            case SDL_KEYDOWN:
                switch (event.key.keysym.sym) {
                    case SDLK_w:
                    case SDLK_a:
                    case SDLK_d:
                    case SDLK_s:
                        camera()->handleInput(
                            CameraInput::Translate::make(event.key.keysym.sym),
                            dt
                        );
                        break;
                    case SDLK_f:
                        auto const currentValue = SDL_ShowCursor(SDL_QUERY);
                        SDL_ShowCursor(currentValue == SDL_ENABLE ? SDL_DISABLE : SDL_ENABLE);
                        break;
                }
            case SDL_MOUSEMOTION:
                auto const isTrackingMoution = SDL_ShowCursor(SDL_QUERY) == SDL_DISABLE;
                if (isTrackingMoution) {
                    // Some strange behavior when cursor is switched it reports huge delta
                    if (std::abs(event.motion.xrel) > (int32_t)resolution.second
                    || std::abs(event.motion.yrel) > (int32_t)resolution.first) {
                        break;
                    }
                    camera()->handleInput(
                        CameraInput::Rotate {
                            .delta = std::make_pair(
                                event.motion.yrel,
                                event.motion.xrel
                            )
                        },
                        dt
                    );
                }
                break;
            }
        }
    }, event);
}

void gl::Renderer::update(MeshData const &data, float dt) {
    static float time = 0.f;
    time += dt;

    auto transformMatrix = ml::scaleMatrix(1.f, 1.f, 1.f, 1.f);
    transformMatrix = ml::matMul(
        ml::rotateAroundPoint({0, 0, 0}, {0, 1, 0}, 0.f),
        transformMatrix
    );
    transformMatrix = ml::matMul(
        ml::translationMatrix(0, 0, -5.f),
        transformMatrix
    );

    gl::materialShader->setUniform(
        "transforms", 
        attributes::Transforms(
            transformMatrix,
            camera()->getViewTransformation(),
            camera()->getScenePerspectiveProjectionMatrix()
        )
    );

    gl::outlineShader->setUniform(
        "transforms", 
        attributes::Transforms(
            ml::matMul(transformMatrix, ml::scaleMatrix(1.1f, 1.1f, 1.1f)),
            camera()->getViewTransformation(),
            camera()->getScenePerspectiveProjectionMatrix()
        )
    );

    light[0].position = camera()->getOrigin();
    gl::materialShader->setUniform("light", light);

     auto const &cameraPosition = camera()->getOrigin();
     for (auto const &[nodeId, node] : scene->nodes) {
        node->addComponent<scene::ShaderInfoComponent>(
            InstanceIdGenerator<scene::ShaderInfoComponent>::getInstanceId(),
            std::unordered_map<std::string, attributes::UniformCases>({
                std::make_pair<std::string, attributes::UniformCases>(
                    "cameraPos",
                    attributes::Vec3({cameraPosition.x, cameraPosition.y, cameraPosition.z})
                )
            })
        );
     }
}

void gl::Renderer::render() const {
    glClearColor(0.5f, 0.1f, 0.3f, 1.0f);
    glClearStencil(0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    
    // Switches for testing
    // shader->use();
    // mockRenderObject.render();
    // mockRenderScene.render();
    renderScene.render();

    SDL_GL_SwapWindow(config->window.get());
}