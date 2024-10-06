#ifndef CLIPPING_H
#define CLIPPING_H

#include "vector.h"
#include "triangle.h"

#define NUM_PLANES 6
#define MAX_NUM_POLY_VERTICES 10
#define MAX_NUM_POLY_TRIANGLES (MAX_NUM_POLY_VERTICES - 2)

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

typedef struct {
    vec3_t vertices[MAX_NUM_POLY_VERTICES];
    int num_vertices;
} polygon_t;

polygon_t create_polygon_from_triangles(
    vec3_t v0,
    vec3_t v1,
    vec3_t v2
);

extern plane_t frustum_planes[NUM_PLANES];

void clip_polygon(polygon_t *polygon);
void init_frustum_planes(float fov, float z_near, float z_far, float aspect_ratioc);
void triangles_from_polygon(polygon_t *polygon, triangle_t *triangles, int *num_triangles);

#endif