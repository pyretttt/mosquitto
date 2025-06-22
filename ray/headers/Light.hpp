#pragma once

#include <variant>

#include "MathUtils.hpp"


struct LightSource {
    ml::Vector3f position;
    ml::Vector3f direction;
    
    ml::Vector3f ambient;
    ml::Vector3f diffuse;
    ml::Vector3f specular;
    
    float cutoffRadians;

    float attenuanceConstant;
    float attenuanceLinear;
    float attenuanceQuadratic;
};