#include <memory>

#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"

#include "Model.hpp"

Model Model::assimpImport(std::filesystem::path path) {
    auto model = Model(path);
    Assimp::Importer importer;
    auto scene = std::make_unique(importer.ReadFile());
}