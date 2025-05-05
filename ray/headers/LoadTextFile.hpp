#pragma once

#include <string>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace utils {

inline std::string loadTextFile(std::filesystem::path path) {
     std::ifstream file = std::ifstream(path.string());
     if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path.string());
     }

     std::stringstream buffer;
     buffer << file.rdbuf();
     return buffer.str();
}

}