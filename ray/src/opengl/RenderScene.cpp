#include "opengl/RenderScene.hpp"


namespace {
    auto makePbrs(
        scene::ScenePtr scene,
        gl::Configuration configuration
    ) noexcept -> decltype(auto) {
        std::vector<gl::RenderObjectInfo> result;

        for (auto const &[id, node] : scene->nodes) {
            for (auto &mesh : node->meshes) {
                result.emplace_back(
                    id,
                    gl::RenderObject<attributes::AssimpVertex>(
                        configuration,
                        mesh,
                        scene->materials.at(mesh->material)
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
        std::unordered_map<size_t, gl::Material> result;
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
                    id, 
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
    gl::Configuration configuration
)
    : scene(scene)
    , configuration(configuration) {}

