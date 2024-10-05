#include <math.h>

#include "clipping.h"

plane_t frustum_planes[NUM_PLANES];

void init_frustum_planes(float fov, float z_near, float z_far) {
    float cos_half_fov = cos(fov / 2);
    float sin_half_fov = sin(fov / 2);

    vec3_t origin = {0, 0, 0};
    frustum_planes[UP_FRUSTUM_PLANE].point = origin;
    frustum_planes[UP_FRUSTUM_PLANE].normal = (vec3_t){
        0, -cos_half_fov, sin_half_fov
    };

    frustum_planes[DOWN_FRUSTUM_PLANE].point = origin;
    frustum_planes[UP_FRUSTUM_PLANE].normal = (vec3_t){
        0, cos_half_fov, sin_half_fov
    };

    frustum_planes[LEFT_FRUSTUM_PLANE].point = origin;
    frustum_planes[LEFT_FRUSTUM_PLANE].normal = (vec3_t){
        cos_half_fov, 0, sin_half_fov
    };

    frustum_planes[RIGHT_FRUSTUM_PLANE].point = origin;
    frustum_planes[RIGHT_FRUSTUM_PLANE].normal = (vec3_t){
        -cos_half_fov, 0, sin_half_fov
    };

    frustum_planes[NEAR_FRUSTUM_PLANE].point = (vec3_t){0, 0, z_near};
    frustum_planes[NEAR_FRUSTUM_PLANE].normal = (vec3_t){
        0, 0, 1
    };

    frustum_planes[FAR_FRUSTUM_PLANE].point = (vec3_t){0, 0, z_far};
    frustum_planes[FAR_FRUSTUM_PLANE].normal = (vec3_t){
       0, 0, -1
    };
}

float distance_from_plane(plane_t plane, )