#pragma once

#include <string>

#include "glad/gl.h"

struct Shader {
    Shader(std::string vertexPath, std::string fragmentPath);

    void use();
    
    template <typename V>
    void setUniform(std::string const &name, V value) const;
    unsigned int id;
};

template <typename V>
void Shader::setUniform(std::string const &name, V value) const {
    if constexpr (std::is_same<V, float>::value) {
        glUniform1f(glGetUniformLocation(id, name.c_str()), value);
    } else if constexpr (std::is_same<V, int>::value) {
        glUniform1i(glGetUniformLocation(id, name.c_str()), value);
    } else if constexpr (std::is_same<V, bool>::value) {
        glUniform1i(glGetUniformLocation(id, name.c_str()), reinterpret_cast<int>(value));
    }
}
