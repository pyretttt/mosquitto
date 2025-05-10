#include <filesystem>
#include <array>
#include <vector>

#include "GL/glew.h"
#include "SDL_opengl.h"
#include "SDL.h"

#include "Attributes.hpp"
#include "opengl/Renderer.hpp"
#include "sdlUtils.hpp"
#include "LoadTextFile.hpp"
#include "opengl/RenderObject.hpp"

namespace fs = std::filesystem;

namespace {
    std::vector<attributes::PositionWithColor> vertexArray = {
        attributes::PositionWithColor {
            .position = {0.5f, 0.5f, 0.0f},
            .color = {0.5f, 0.3f, 0.9f, 1.0f}
        },
        attributes::PositionWithColor {
            .position = {0.5f, -0.5f, 0.0f},
            .color = {0.7f, 0.2f, 0.6f, 1.0f}
        },
        attributes::PositionWithColor {
            .position = {-0.5f, -0.5f, 0.0f},
            .color = {0.5f, 0.9f, 0.4f, 1.0f}
        },
        attributes::PositionWithColor {
            .position = {-0.5f, 0.5f, 0.0f},
            .color = {0.5f, 0.3f, 0.3f, 1.0f}
        }
        // attributes::Vec3 {.val = {0.5f, 0.5f, 0.0f}},
        // attributes::Vec3 {.val = {0.5f, -0.5f, 0.0f}},
        // attributes::Vec3 {.val = {-0.5f, -0.5f, 0.0f}},
        // attributes::Vec3 {.val = {-0.5f, 0.5f, 0.0f}},
    };

    gl::EBO vertexArrayIndices = {
        0, 1, 3,
        1, 2, 3 
    };

    gl::Configuration config = gl::Configuration {
        .polygonMode = gl::Configuration::PolygonMode {
            .face = GL_FRONT,
            .mode = GL_FILL
        }
    };

    gl::RenderObject renderObject = gl::RenderObject<attributes::PositionWithColor>(
        vertexArray,
        vertexArrayIndices,
        config,
        fs::path("shaders").append("vertex.vs"),
        fs::path("shaders").append("fragment.fs"),
        false,
        true
    );
}

gl::Renderer::Renderer(
    std::shared_ptr<GlobalConfig> config
)
    : config(config)
    , resolution(config->windowSize.value()) {        
}

gl::Renderer::~Renderer() {
    SDL_GL_DeleteContext(glContext);
}

void gl::Renderer::prepareViewPort() {
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

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

    renderObject.prepare();
}

void gl::Renderer::processInput(Event) {

}

void gl::Renderer::update(MeshData const &data, float dt) {
    static float red = 0.f;
    red += dt / 10000;
    red = red - static_cast<int>(red);
    renderObject.setUniform("ourColor", attributes::Vec4 {.val = {red, 0.3f, 0.4f, 1.0f}});
}

void gl::Renderer::render() const {
    glClearColor(0.5f, 0.1f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    renderObject.render();

    SDL_GL_SwapWindow(config->window.get());
}