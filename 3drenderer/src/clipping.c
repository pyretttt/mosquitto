#include <math.h>

#include "clipping.h"

plane_t frustum_planes[NUM_PLANES];

void init_frustum_planes(float fov, float z_near, float z_far, float aspect_ratio) {
    float cos_half_fov_y = cos(fov / 2);
    float sin_half_fov_y = sin(fov / 2);

    float omega = atan(aspect_ratio * tan(fov / 2));
    float cos_half_fov_x = cos(omega);
    float sin_half_fov_x = sin(omega);

    vec3_t origin = {0, 0, 0};
    frustum_planes[UP_FRUSTUM_PLANE].point = origin;
    frustum_planes[UP_FRUSTUM_PLANE].normal = (vec3_t){
        0, -cos_half_fov_y, sin_half_fov_y
    };

    frustum_planes[DOWN_FRUSTUM_PLANE].point = origin;
    frustum_planes[DOWN_FRUSTUM_PLANE].normal = (vec3_t){
        0, cos_half_fov_y, sin_half_fov_y
    };

    frustum_planes[LEFT_FRUSTUM_PLANE].point = origin;
    frustum_planes[LEFT_FRUSTUM_PLANE].normal = (vec3_t){
        cos_half_fov_x, 0, sin_half_fov_x
    };

    frustum_planes[RIGHT_FRUSTUM_PLANE].point = origin;
    frustum_planes[RIGHT_FRUSTUM_PLANE].normal = (vec3_t){
        -cos_half_fov_x, 0, sin_half_fov_x
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
    vec3_t v2,
    tex2_t t0,
    tex2_t t1,
    tex2_t t2
) {
    polygon_t p = {
        .num_vertices = 3,
        .texcoords = {t0, t1, t2},
        .vertices = {v0, v1, v2},
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
    vec3_t delta = vec3_mul(vec3_sub(v1, v0), t);
    return vec3_add(v0, delta);
}

static inline tex2_t interpolate_uv(vec3_t v0, vec3_t v1, tex2_t t0, tex2_t t1, plane_t plane) {
    float v0_dist = distance_from_plane(v0, plane);
    float v1_dist = distance_from_plane(v1, plane);

    float t = v0_dist / (v0_dist - v1_dist);

    float new_u = t0.u + t * (t1.u - t0.u);
    float new_v = t0.v + t * (t1.v - t0.v);
    return (tex2_t){.u = new_u, .v = new_v};
}

static void clip_polygon_against_plane(polygon_t *polygon, int frustum_plane) {
    plane_t plane = frustum_planes[frustum_plane];
    vec3_t inside[MAX_NUM_POLY_VERTICES];
    tex2_t inside_texcoords[MAX_NUM_POLY_VERTICES];
    int insideCount = 0;
    vec3_t *current_vertex = &polygon->vertices[0];
    tex2_t *current_textcoord = &polygon->texcoords[0];
    vec3_t *previous_vertex = &polygon->vertices[polygon->num_vertices - 1];
    tex2_t *previous_textcoord = &polygon->texcoords[polygon->num_vertices - 1];

    float previous_distance = distance_from_plane(*previous_vertex, plane);

    while (current_vertex != &polygon->vertices[polygon->num_vertices]) {
        float current_distance = distance_from_plane(*current_vertex, plane);
        // if moving from inside to outside or vice versa, add intersection point
        if (current_distance * previous_distance < 0) {
            vec3_t intersection = intersection_point(*current_vertex, *previous_vertex, plane);

            tex2_t interpolated_coord = interpolate_uv(
                *current_vertex, 
                *previous_vertex, 
                *current_textcoord, 
                *previous_textcoord, 
                plane
            );

            inside[insideCount] = intersection;
            inside_texcoords[insideCount] = interpolated_coord;
            insideCount++;
        }
        // also check wether current point inside
        if (current_distance > 0) {
            inside[insideCount] = *current_vertex;
            inside_texcoords[insideCount] = *current_textcoord;
            insideCount++;
        }

        previous_distance = current_distance;
        previous_vertex = current_vertex;
        previous_textcoord = current_textcoord;
        current_vertex++;
        current_textcoord++;
    }
    
    for (int i = 0; i < insideCount; ++i) {
        polygon->vertices[i] = inside[i];
        polygon->texcoords[i] = inside_texcoords[i];
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

void triangles_from_polygon(polygon_t *polygon, triangle_t *triangles, int *num_triangles) {
    *num_triangles = polygon->num_vertices - 2;

    for (int i = 0; i < (polygon->num_vertices - 2); i++) {
		int i1 = i + 1;
		int i2 = i + 2;
		
		triangles[i].points[0] = vec4_from_vec3(polygon->vertices[0]);
		triangles[i].tex_coords[0] = polygon->texcoords[0];
		triangles[i].points[1] = vec4_from_vec3(polygon->vertices[i1]);
		triangles[i].tex_coords[1] = polygon->texcoords[i1];
		triangles[i].points[2] = vec4_from_vec3(polygon->vertices[i2]);
		triangles[i].tex_coords[2] = polygon->texcoords[i2];
	}
}