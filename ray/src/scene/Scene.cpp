#include <memory>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <variant>

#include "assimp/Importer.hpp"
#include "assimp/DefaultLogger.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"

#include "scene/Scene.hpp"
#include "scene/Node.hpp"
#include "scene/Mesh.hpp"
#include "scene/Attachment.hpp"
#include "Core.hpp"
#include "scene/MeshComponent.hpp"

using namespace scene;

namespace {
    constexpr size_t numVerticesInFace = 3;

    std::shared_ptr<scene::Mesh<>> makeMesh(
        aiMesh *mesh, 
        aiScene const *scene, 
        std::unordered_map<MaterialId, scene::MaterialPtr> materialsMap
    ) {
        std::vector<attributes::Cases> vertices;
        vertices.reserve(mesh->mNumVertices);
        std::vector<unsigned int> verticesArrayIndices;
        verticesArrayIndices.reserve(mesh->mNumFaces * numVerticesInFace);

        for (size_t i = 0; i < mesh->mNumVertices; i++) {
            auto pos = mesh->mVertices[i];
            auto normal = mesh->mNormals[i];

            aiVector3D tex = {0.f, 0.f, 0.f};
            if (mesh->mMaterialIndex >= 0) {
                tex = mesh->mTextureCoords[0][i];
            }
            vertices.push_back(
                attributes::Cases(
                    attributes::MaterialVertex { 
                        .position = {pos.x, pos.y, pos.z},
                        .normal = {normal.x, normal.y, normal.z},
                        .tex = {tex.x, tex.y}
                    }
                )
            );
        }

        for (size_t i = 0; i < mesh->mNumFaces; i++) {
            auto &face = mesh->mFaces[i];
            for (size_t j = 0; j < face.mNumIndices; j++) {
                verticesArrayIndices.push_back(face.mIndices[j]);
            }
        }

        AttachmentCases attachment = mesh->mMaterialIndex >= 0
            ? AttachmentCases(scene::MaterialAttachment {
                .material = materialsMap.at(mesh->mMaterialIndex), 
                .id = mesh->mMaterialIndex
            })
            : AttachmentCases(std::monostate());
        
        return std::make_shared<scene::Mesh<>>(
            std::move(vertices),
            std::move(verticesArrayIndices),
            attachment,
            static_cast<MeshId>(InstanceIdGenerator<scene::Mesh<>>::getInstanceId())
        );
    }

    scene::NodePtr makeNode(
        aiScene const *scene,
        aiNode *node,
        std::unordered_map<MaterialId, scene::MaterialPtr> const &materialsMap 
    ) {
        std::vector<std::shared_ptr<scene::Mesh<>>> meshes;
        for (size_t i = 0; i < node->mNumMeshes; i++) {
            meshes.emplace_back(
                makeMesh(
                    scene->mMeshes[node->mMeshes[i]],
                    scene,
                    materialsMap
                )
            );
        }

        auto nodePtr = std::make_shared<scene::Node>(
            static_cast<NodeId>(InstanceIdGenerator<scene::Node>::getInstanceId())
        );
        nodePtr->addComponent<scene::MeshComponent<>>(
            InstanceIdGenerator<scene::MeshComponent<>>::getInstanceId(),
            std::move(meshes)
        );
        return nodePtr;
    }

    std::vector<scene::NodePtr> genNodes(
        aiNode *node,
        aiScene const *scene, 
        std::unordered_map<MaterialId, scene::MaterialPtr> const &materialsMap
    ) {
        auto rootNode = makeNode(scene, node, materialsMap);
        std::vector<scene::NodePtr> children;
        for (size_t i = 0; i < node->mNumChildren; i++) {
            auto childNodes = genNodes(node->mChildren[i], scene, materialsMap);
            std::for_each(childNodes.begin(), childNodes.end(), [&rootNode, &children](scene::NodePtr &child){
                child->parent = rootNode;
                children.push_back(child);
            });
        }
        
        std::vector<scene::NodePtr> result = {rootNode};
        std::move(children.begin(), children.end(), std::back_inserter(result));
        return result;
    }
}

std::vector<scene::TexturePtr> loadTextures(
    std::filesystem::path path, 
    aiMaterial *material, 
    aiTextureType type,
    std::unordered_map<std::string, TexturePtr> &cache
) {
    std::vector<scene::TexturePtr> textures;

    for (size_t i = 0; i < material->GetTextureCount(type); i++) {
        aiString subPath;
        material->GetTexture(type, i, &subPath);
        std::filesystem::path texturePath = path.append(subPath.C_Str());

        if (!cache.contains(static_cast<TexturePath>(texturePath.string()))) {
            cache[static_cast<TexturePath>(texturePath.string())] = std::make_shared<scene::TexData>(
                loadTextureData(texturePath)
            );
        }

        textures.emplace_back(cache.at(static_cast<TexturePath>(texturePath.string())));
    }

    return textures;
}

Scene::Scene(
    std::unordered_map<NodeId, NodePtr> nodes,
    std::unordered_map<MaterialId, MaterialPtr> materials,
    std::unordered_map<TexturePath, TexturePtr> textures
) 
    : nodes(std::move(nodes))
    , materials(std::move(materials))
    , textures(std::move(textures)) {}


Scene Scene::assimpImport(std::filesystem::path path) {
    Assimp::Importer importer;
    Assimp::DefaultLogger::create();
    
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

    std::unordered_map<TexturePath, TexturePtr> texturesMap;
    std::unordered_map<MaterialId, scene::MaterialPtr> materialsMap;
    for (size_t i = 0; i < scene->mNumMaterials; i++) {
        aiMaterial *material = scene->mMaterials[i];
        float shiness;
        material->Get(AI_MATKEY_SHININESS, shiness);

        aiColor3D ambient;
        material->Get(AI_MATKEY_COLOR_AMBIENT, ambient);

        std::vector<TexturePtr> ambients = loadTextures(path.parent_path(), material, aiTextureType_AMBIENT, texturesMap);
        std::vector<TexturePtr> diffuse = loadTextures(path.parent_path(), material, aiTextureType_DIFFUSE, texturesMap);
        std::vector<TexturePtr> specular = loadTextures(path.parent_path(), material, aiTextureType_SPECULAR, texturesMap);
        std::vector<TexturePtr> normals = loadTextures(path.parent_path(), material, aiTextureType_NORMALS, texturesMap);

        materialsMap[i] = std::make_shared<Material>(
            attributes::Vec3({ambient.r, ambient.g, ambient.b}),
            0.5, //shiness,
            ambients.size() > 0 ? ambients : diffuse,
            diffuse,
            specular,
            normals
        );
    }

    auto nodes = genNodes(scene->mRootNode, scene, materialsMap);
    std::unordered_map<NodeId, scene::NodePtr> nodesMap;
    std::transform(nodes.begin(), nodes.end(), std::inserter(nodesMap, nodesMap.end()), [](NodePtr &node) {
        return std::make_pair(node->identifier, node);
    });

    return Scene(
        std::move(nodesMap),
        std::move(materialsMap),
        std::move(texturesMap)
    );
}

