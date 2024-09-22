#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <stdlib.h>

#include "vector.h"
#include "texture.h"

typedef struct {
    int a, b, c;
    tex2_t a_uv, b_uv, c_uv;
    uint32_t color;
} face_t;

typedef struct {
    vec2_t points[3];
    tex2_t tex_coords[3];
    uint32_t color;
    float avg_depth;
} triangle_t;

void draw_filled_triangle(int x0, int y0, int x1, int y1, int x2, int y2, uint32_t color);
void draw_textured_triangle(
    int x0, int y0, float u0, float v0, 
    int x1, int y1, float u1, float v1,
    int x2, int y2, float u2, float v2, 
    uint32_t *texture
);

void fill_flat_bottom_triangle(int x0, int y0, int x1, int y1, int x2, int y2, uint32_t color);
void fill_flat_top_triangle(int x0, int y0, int x1, int y1, int x2, int y2, uint32_t color);

#endif