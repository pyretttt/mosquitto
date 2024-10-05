#ifndef CLIPPING_H
#define CLIPPING_H

#include "vector.h"

#define NUM_PLANES 6

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

extern plane_t frustum_planes[NUM_PLANES];

void init_frustum_planes(float fov, float z_near, float z_far);

#endif