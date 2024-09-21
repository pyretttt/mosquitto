#include "matrix.h"
#include <math.h>

mat4_t mat4_idenity(void) {
    static mat4_t res = {
        .m = {
            {1, 0, 0, 0},
            {0, 1, 0, 0},
            {0, 0, 1, 0},
            {0, 0, 0, 1},
        }
    };

    return res;
}

mat4_t make_scale_matrix(float sx, float sy, float sz) {
    mat4_t res = mat4_idenity();
    res.m[0][0] = sx;
    res.m[1][1] = sy;
    res.m[2][2] = sz;

    return res;
}

mat4_t make_translation_matrix(float tx, float ty, float tz) {
    mat4_t res = mat4_idenity();
    res.m[0][3] = tx;
    res.m[1][3] = ty;
    res.m[2][3] = tz;

    return res;
}

mat4_t make_rotation_matrix_z(float r) {
    mat4_t res = mat4_idenity();
    float c = cos(r);
    float s = sin(r);

    res.m[0][0] = c;
    res.m[0][1] = -s;
    res.m[1][0] = s;
    res.m[1][1] = c ;
    return res;
}

mat4_t make_rotation_matrix_y(float r) {
    mat4_t res = mat4_idenity();
    float c = cos(r);
    float s = sin(r);

    res.m[0][0] = c;
    res.m[0][2] = s;
    res.m[2][0] = -s;
    res.m[2][2] = c ;
    return res;
}
mat4_t make_rotation_matrix_x(float r) {
    mat4_t res = mat4_idenity();
    float c = cos(r);
    float s = sin(r);

    res.m[1][1] = c;
    res.m[1][2] = -s;
    res.m[2][1] = s;
    res.m[2][2] = c ;
    return res;
}

// aspect_ratio is w/h
mat4_t make_projection_matrix(float fov, float aspect_ratio, float znear, float zfar) {
    mat4_t mat = {0};
    float tangent = tanf(fov / 2);
    mat.m[0][0] =  1 / aspect_ratio * tangent;
    mat.m[1][1] =  1 / tangent;
    mat.m[2][2] =  zfar / (zfar - znear);
    mat.m[2][3] =  - (zfar * znear) / (zfar - znear);
    mat.m[3][2] =  1.0f;

    return mat;
}

vec4_t mat4_mul_vec4_project(mat4_t mat_proj, vec4_t v) {
    vec4_t res = mul_vec4(mat_proj, v);

    if (res.w != 0.0) {
        res.x /= res.w;
        res.y /= res.w;
        res.z /= res.w;
    }

    return res;
}

vec4_t mul_vec4(mat4_t m, vec4_t v) {
    vec4_t result;
    result.x = m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z + m.m[0][3] * v.w;
    result.y = m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z + m.m[1][3] * v.w;
    result.z = m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z + m.m[2][3] * v.w;
    result.w = m.m[3][0] * v.x + m.m[3][1] * v.y + m.m[3][2] * v.z + m.m[3][3] * v.w;
    return result;
}

mat4_t mat4_mul(mat4_t a, mat4_t b) {
    mat4_t res = {0};
    for (int col = 0; col < 4; col++) {
        for (int row = 0; row < 4; row++) {
            // for (int k = 0; k < 4; k++) {
            //     res.m[row][col] += a[row][k] * b[k][col];
            // }
            res.m[row][col] = a.m[row][0] * b.m[0][col]
                + a.m[row][1] * b.m[1][col]
                + a.m[row][2] * b.m[2][col]
                + a.m[row][3] * b.m[3][col];
        }
    }
    return res;
}
