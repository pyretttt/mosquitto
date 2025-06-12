#include "opengl/RenderScene.hpp"
#include "scene/Material.hpp"
#include "scene/Node.hpp"

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
}

void gl::RenderScene::render() const {
    pbrShader->use();
    for (auto const &renderObjectInfo : pbrs) {
        auto const &material = renderObjectInfo.renderObject.material;
        pbrShader->setMaterialSamplers(material);
        renderObjectInfo.renderObject.render();
    }
}