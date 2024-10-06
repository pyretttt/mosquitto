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
    frustum_planes[DOWN_FRUSTUM_PLANE].normal = (vec3_t){
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

polygon_t create_polygon_from_triangles(
    vec3_t v0,
    vec3_t v1,
    vec3_t v2
) {
    polygon_t p = {
        .num_vertices = 3,
        .vertices = {v0, v1, v2}
    };
    return p;
}

static inline float distance_from_plane(vec3_t point, plane_t plane) {
    vec3_t diff_vector = vec3_sub(point, plane.point);
    return vec3_dot_product(plane.normal, diff_vector);
}

static inline vec3_t intersection_point(vec3_t v0, vec3_t v1, plane_t plane) {
    float v0_dist = distance_from_plane(v0, plane);
    float v1_dist = distance_from_plane(v1, plane);

    float t = v0_dist / (v0_dist - v1_dist);
    vec3_t delta = vec3_mul(vec3_sub(v0, v1), t);
    return vec3_add(v1, delta);
}

static void clip_polygon_against_plane(polygon_t *polygon, int frustum_plane) {
    plane_t plane = frustum_planes[frustum_plane];
    vec3_t inside[MAX_NUM_POLY_VERTICES];
    int insideCount = 0;
    vec3_t *current_vertex = &polygon->vertices[0];
    vec3_t *previous_vertex = &polygon->vertices[polygon->num_vertices - 1];

    float previous_distance = distance_from_plane(*previous_vertex, plane);

    while (current_vertex != &polygon->vertices[polygon->num_vertices]) {
        float current_distance = distance_from_plane(*current_vertex, plane);
        // if moving from inside to outside or vice versa, add intersection point
        if (current_distance * previous_distance < 0) {
            vec3_t intersection = intersection_point(*current_vertex, *previous_vertex, plane);
            inside[insideCount++] = intersection;
        }
        // also check wether current point inside
        if (current_distance > 0) {
            inside[insideCount++] = *current_vertex;
        }

        previous_distance = current_distance;
        previous_vertex = current_vertex;
        current_vertex++;
    }
    
    for (int i = 0; i < insideCount; ++i) {
        polygon->vertices[i] = inside[i];
    }
    polygon->num_vertices = insideCount;
}

void clip_polygon(polygon_t *polygon) {
    clip_polygon_against_plane(polygon, LEFT_FRUSTUM_PLANE);
    clip_polygon_against_plane(polygon, RIGHT_FRUSTUM_PLANE);
    clip_polygon_against_plane(polygon, UP_FRUSTUM_PLANE);
    clip_polygon_against_plane(polygon, DOWN_FRUSTUM_PLANE);
    clip_polygon_against_plane(polygon, NEAR_FRUSTUM_PLANE);
    clip_polygon_against_plane(polygon, FAR_FRUSTUM_PLANE);
}
