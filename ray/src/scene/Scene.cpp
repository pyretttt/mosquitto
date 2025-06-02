#include <memory>
#include <algorithm>
#include <iterator>

#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"

#include "scene/Scene.hpp"
#include "scene/Mesh.hpp"
#include "Core.hpp"

using namespace scene;

namespace {
    scene::MeshPtr makeMesh(aiMesh *mesh) {
        std::vector<attributes::AssimpVertex> vertices;
        std::vector<unsigned int> verticesArrayIndices;
        std::vector<scene::TextureIdentifier> textures;

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
                verticesArrayIndices.push_back(face.mIndices[j]);
            }
        }
        
        return std::make_shared<scene::Mesh<attributes::AssimpVertex>>(
            std::move(vertices),
            std::move(verticesArrayIndices),
            std::move(textures),
            InstanceIdGenerator<scene::Mesh<attributes::AssimpVertex>>::getInstanceId()
        );
    }

    scene::NodePtr makeNode(aiScene const *scene, aiNode *node) {
        std::vector<scene::MeshPtr> meshes;
        for (size_t i = 0; i < node->mNumMeshes; i++) {
            meshes.emplace_back(makeMesh(scene->mMeshes[i]));
        }

        return std::make_shared<scene::Node>(
            InstanceIdGenerator<scene::Node>::getInstanceId(),
            std::move(meshes)
        );
    }

    std::vector<scene::NodePtr> genNodes(aiNode *node, aiScene const *scene) {
        auto rootNode = makeNode(scene, node);
        std::vector<scene::NodePtr> children;
        for (size_t i = 0; i < node->mNumChildren; i++) {
            auto childNodes = genNodes(node->mChildren[i], scene);
            std::for_each(childNodes.begin(), childNodes.end(), [&rootNode, &children](scene::NodePtr &child){
                child->parent = rootNode;
                children.push_back(child);
            });
        }
        
        std::vector<scene::NodePtr> result = {rootNode};
        std::move(children.begin(), children.end(), std::back_inserter(result));
        return std::move(result);
    }
}

Scene::Scene(
    std::unordered_map<size_t, NodePtr> nodes,
    std::unordered_map<size_t, TexturePtr> textures
) 
    : nodes(std::move(nodes))
    , textures(std::move(textures)) {}

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

    auto nodes = genNodes(scene->mRootNode, scene);
    std::unordered_map<size_t, scene::NodePtr> nodesMap;
    std::transform(nodes.begin(), nodes.end(), std::inserter(nodesMap, nodesMap.end()), [](auto const &node) {
        return std::make_pair(node->identifier, node);
    });

    std::unordered_map<size_t, scene::TexturePtr> textureMap;
    return Scene(
        std::move(nodesMap),
        std::move(textureMap)
    );
}

