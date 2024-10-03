#ifndef CLIPPING_H
#define CLIPPING_H

#include "vector.h"

enum {
    LEFT_FRUSTUM_PLANE,
    RIGHT_FRUSTUM_PLANE,
    UP_FRUSTUM_PLANE,
    DOWN_FRUSTUM_PLANE,
    NEAR_FRUSTUM_PLANE,
    FAR_FRUSTUM_PLANE
};

typedef struct {
    vec3_t normal;
    vec3_t point;
} plane_t;

#endif