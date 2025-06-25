#pragma once

#include <variant>

#include "MathUtils.hpp"


struct LightSource {
    ml::Vector3f position;
    ml::Vector3f spotDirection;
    
    ml::Vector3f ambient;
    ml::Vector3f diffuse;
    ml::Vector3f specular;
    
    float cutoffRadians;
    float cutoffDecayRadians;

    float attenuanceConstant;
    float attenuanceLinear;
    float attenuanceQuadratic;
};