#include <memory>

#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"

#include "Scene.hpp"
#include "opengl/MeshNode.hpp"
#include "opengl/glCommon.hpp"

namespace {
    std::vector<Scene::Mesh> genMeshes(aiNode *node, aiScene const *scene) {
        std::vector<aiMesh *> meshes;
        for (size_t i = 0; i < node->mNumMeshes; i++) {
            aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
            meshes.emplace_back(makeMesh(mesh));
        }

        for (size_t i = 0; i < node->mNumChildren; i++) {
            auto childMeshes = genMeshes(node->mChildren[i], scene);
            if (!meshes.empty()) {
                std::for_each(childMeshes.begin(), childMeshes.end(), [](Scene::Mesh &mesh){
                    mesh.parent = mesh
                    meshes.at(0)
                });
            }
        }
    }

    Scene::Mesh makeMesh(aiMesh *mesh) {
        std::vector<attributes::AssimpVertex> vertices;
        gl::EBO ebo;
        std::vector<size_t> texture;

        for (size_t i = 0; i < mesh->mNumVertices; i++) {
            auto pos = mesh->mVertices[i];
            auto normal = mesh->mNormals[i];

            aiVector3D tex = {0.f, 0.f, 0.f};
            if (mesh->mMaterialIndex >= 0) {
                tex = mesh->mTextureCoords[0][i];
            }
            vertices.push_back(
                attributes::AssimpVertex { 
                    .position = {pos.x, pos.y, pos.z},
                    .normal = {normal.x, normal.y, normal.z},
                    .tex = {tex.x, tex.y}
                }
            );
        }

        for (size_t i = 0; i < mesh->mNumFaces; i++) {
            auto face = mesh->mFaces[i];
            for (size_t j = 0; j <face.mNumIndices; j++) {
                ebo.push_back(face.mIndices[j]);
            }
        }

        return gl::MeshNode<attributes::AssimpVertex>(
            std::move(vertices),
            std::move(ebo)
        );
    }
}

Scene Scene::assimpImport(std::filesystem::path path) {
    // auto model = Model(path);
    Assimp::Importer importer;
    auto scene = importer.ReadFile(
        path.string(), 
        aiProcess_Triangulate
        | aiProcess_FlipUVs
        | aiProcess_GenNormals
        | aiProcess_JoinIdenticalVertices
    );
    if (
        !scene
        || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE
        || !scene->mRootNode
    ) {
        throw "Failed to load scene";
    }


}

