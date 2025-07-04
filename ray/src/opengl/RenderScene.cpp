#include <memory>
#include <filesystem>
#include <variant>

#include "opengl/RenderScene.hpp"
#include "scene/Material.hpp"
#include "scene/Node.hpp"
#include "scene/ShaderInfoComponent.hpp"
#include "opengl/Uniforms.hpp"
namespace {
    auto makePbrs(
        scene::ScenePtr scene,
        gl::Configuration configuration,
        std::unordered_map<scene::MaterialId, gl::Material> const &materials
    ) -> decltype(auto) {
        std::vector<gl::RenderObjectInfo> result;

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
                    gl::RenderObject<attributes::AssimpVertex>(
                        configuration,
                        mesh,
                        materials.at(materialId)
                    )
                );
            }
        }

        return result;
    }

    auto toGlTexture(scene::TexturePtr texture) -> decltype(auto) {
        return std::make_shared<gl::Texture>(texture);
    }

    auto makeMaterials(scene::ScenePtr scene) -> decltype(auto) {
        std::unordered_map<scene::MaterialId, gl::Material> result;
        for (auto const &[id, material] : scene->materials) {
            std::vector<gl::TexturePtr> ambient, specular, diffuse, normals;
            std::transform(
                material->ambient.begin(), 
                material->ambient.end(), 
                std::back_inserter(ambient),
                toGlTexture
            );
            std::transform(
                material->specular.begin(), 
                material->specular.end(), 
                std::back_inserter(specular),
                toGlTexture
            );
            std::transform(
                material->diffuse.begin(), 
                material->diffuse.end(), 
                std::back_inserter(diffuse),
                toGlTexture
            );
            std::transform(
                material->normals.begin(), 
                material->normals.end(), 
                std::back_inserter(normals),
                toGlTexture
            );
            result.emplace(
                std::make_pair(
                    static_cast<scene::MaterialId>(id), 
                    gl::Material {
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
    gl::Configuration configuration,
    ShaderPtr pbrShader
)
    : scene(scene)
    , configuration(configuration)
    , pbrShader(pbrShader) {}

void gl::RenderScene::prepare() {
    this->materials = makeMaterials(scene);
    this->pbrs = makePbrs(scene, configuration, materials);
    pbrShader->setup();

    std::for_each(pbrs.begin(), pbrs.end(), [](gl::RenderObjectInfo &renderObjectInfo) {
        renderObjectInfo.renderObject.prepare();
    });

    outlineShader->setup();
}

void gl::RenderScene::render() const {
    for (auto const &renderObjectInfo : pbrs) {
        pbrShader->use();
        auto const &node = scene->nodes.at(renderObjectInfo.nodeId);
        if (auto component = node->getComponent<scene::ShaderInfoComponent>()) {
            auto shaderInfo = std::static_pointer_cast<scene::ShaderInfoComponent>(component.value());
            for (auto const &[key, attribute] : shaderInfo->uniforms) {
                pbrShader->setUniform(key, attributes::UniformCases(attribute));
            }
        }

        auto const &material = renderObjectInfo.renderObject.material;
        pbrShader->setUniform("material", material);
        renderObjectInfo.renderObject.render();
    }
}