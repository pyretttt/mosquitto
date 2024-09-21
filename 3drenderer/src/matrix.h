#ifndef MATRIX_H
#define MATRIX_H

#include "vector.h"

typedef struct {
    float m[4][4];
} mat4_t;

mat4_t mat4_idenity(void);
mat4_t make_scale_matrix(float sx, float sy, float sz);
mat4_t make_translation_matrix(float tx, float ty, float tz);
mat4_t make_rotation_matrix_z(float r);
mat4_t make_rotation_matrix_y(float r);
mat4_t make_rotation_matrix_x(float r);
mat4_t make_projection_matrix(float fov, float aspect_ratio, float znear, float zfar);

vec4_t mul_vec4(mat4_t m, vec4_t v);
mat4_t mat4_mul(mat4_t a, mat4_t b);
vec4_t mat4_mul_vec4_project(mat4_t mat_proj, vec4_t v);

#endif