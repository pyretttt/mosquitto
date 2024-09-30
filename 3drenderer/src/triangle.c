#include "triangle.h"
#include "display.h"

static void swap(int *a, int *b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

static void fswap(float *a, float *b)
{
    float temp = *a;
    *a = *b;
    *b = temp;
}

void fill_flat_bottom_triangle(int x0, int y0, int x1, int y1, int x2, int y2, uint32_t color)
{
    float inv_slope_left = (float)(x1 - x0) / (y1 - y0);
    float inv_slope_right = (float)(x2 - x0) / (y2 - y0);

    float x_start = x0;
    float x_end = x0;
    for (int y = y0; y <= y2; ++y)
    {
        draw_line(x_start, y, x_end, y, color);
        x_start += inv_slope_left;
        x_end += inv_slope_right;
    }
}

void fill_flat_top_triangle(int x0, int y0, int x1, int y1, int x2, int y2, uint32_t color)
{
    float inv_left_slope = (float)(x2 - x0) / (y2 - y0);
    float inv_right_slope = (float)(x1 - x2) / (y1 - y2);

    float x_start = x2;
    float x_end = x2;

    for (int y = y2; y >= y0; --y)
    {
        draw_line(x_start, y, x_end, y, color);
        x_start -= inv_left_slope;
        x_end -= inv_right_slope;
    }
}

void draw_filled_triangle(int x0, int y0, int x1, int y1, int x2, int y2, uint32_t color)
{
    //     v0
    //   /    \
    //  /      \
    // v1 ----  \
    //   \_      \
    //     \_     \
    //       \_    \
    //         \_   \
    //           \_  \
    //              \_\
    //                v2

    if (y0 > y1) {
        swap(&y0, &y1);
        swap(&x0, &x1);
    }
    if (y1 > y2) {
        swap(&y1, &y2);
        swap(&x1, &x2);
    }
    if (y0 > y1) {
        swap(&y0, &y1);
        swap(&x0, &x1);
    }

    // Avoids zero division exception
    if (y1 == y2) {
        fill_flat_bottom_triangle(x0, y0, x1, y1, x2, y2, color);
    } else if (y0 == y1) {
        fill_flat_top_triangle(x0, y0, x1, y1, x2, y2, color);
    } else {
        int m_y = y1;
        int m_x = x0 + ((float)(x2 - x0) * (m_y - y0)) / (float)(y2 - y0);

        fill_flat_bottom_triangle(x0, y0, x1, y1, m_x, m_y, color);
        fill_flat_top_triangle(x1, y1, m_x, m_y, x2, y2, color);
    }
}

static vec3_t compute_weights(vec2_t a, vec2_t b, vec2_t c, vec2_t p) {
    vec2_t ac = vec2_sub(c, a);
    vec2_t ab = vec2_sub(b, a);
    vec2_t ap = vec2_sub(p, a);
    vec2_t pc = vec2_sub(c, p);
    vec2_t pb = vec2_sub(b, p);

    float area_parallelogram_abc = (ac.x * ab.y - ac.y * ab.x); // || AC x AB ||
    float alpha = (pc.x * pb.y - pc.y * pb.x) / area_parallelogram_abc;
    float beta = (ac.x * ap.y - ac.y * ap.x) / area_parallelogram_abc;
    float gamma = 1 - alpha - beta;

    return (vec3_t){ alpha, beta, gamma };
}

static void draw_texel(
    int x, int y, uint32_t* texture,
    vec4_t point_a, vec4_t point_b, vec4_t point_c,
    tex2_t a_uv, tex2_t b_uv, tex2_t c_uv
) {
    vec2_t a = vec2_from_vec4(point_a);
    vec2_t b = vec2_from_vec4(point_b);
    vec2_t c = vec2_from_vec4(point_c);
    vec3_t weights = compute_weights(
        a,
        b,
        c,
        (vec2_t){x, y}
    );
    float alpha = weights.x;
    float beta = weights.y;
    float gamma = weights.z;

    float w_recip = alpha * (1 / point_a.w) + beta * (1 / point_b.w) + gamma * (1 / point_c.w);
    float u = alpha * (a_uv.u / point_a.w) + beta * (b_uv.u / point_b.w) + gamma * (c_uv.u / point_c.w);
    float v = alpha * (a_uv.v / point_a.w) + beta * (b_uv.v / point_b.w) + gamma * (c_uv.v / point_c.w);
    u /= w_recip;
    v /= w_recip;

    int tex_x = abs((int)(u * texture_width)) % texture_width;
    int tex_y = abs((int)(v * texture_height)) % texture_height;
    uint32_t texel = texture[(texture_width * tex_y + tex_x)];

    draw_pixel(x, y, texel);
}

void draw_textured_triangle(
    int x0, int y0, float z0, float w0, float u0, float v0, 
    int x1, int y1, float z1, float w1, float u1, float v1,
    int x2, int y2, float z2, float w2, float u2, float v2, 
    uint32_t *texture
) {
    //     v0
    //   /    \
    //  /      \
    // v1 ----  \
    //   \_      \
    //     \_     \
    //       \_    \
    //         \_   \
    //           \_  \
    //              \_\
    //                v2
    if (y1 < y0) {
        swap(&y0, &y1);
        swap(&x0, &x1);
        fswap(&u0, &u1);
        fswap(&v0, &v1);
        fswap(&z0, &z1);
        fswap(&w0, &w1);
    }
    if (y2 < y1) {
        swap(&y2, &y1);
        swap(&x2, &x1);
        fswap(&u2, &u1);
        fswap(&v2, &v1);
        fswap(&z2, &z1);
        fswap(&w2, &w1);
    }
    if (y1 < y0) {
        swap(&y0, &y1);
        swap(&x0, &x1);
        fswap(&u0, &u1);
        fswap(&v0, &v1);
        fswap(&z0, &z1);
        fswap(&w0, &w1);
    }
    v0 = 1 - v0;
    v1 = 1 - v1;
    v2 = 1 - v2;

    vec4_t point_a = { x0, y0, z0, w0 };
    vec4_t point_b = { x1, y1, z1, w1 };
    vec4_t point_c = { x2, y2, z2, w2 };
    tex2_t a_uv = {u0, v0};
    tex2_t b_uv = {u1, v1};
    tex2_t c_uv = {u2, v2};

    float inv_slop_1 = y1 != y0
        ? (float)(x1 - x0) / abs(y1 - y0)
        : 0;
    float inv_slop_2 = y2 != y0
        ? (float)(x2 - x0) / abs(y2 - y0)
        : 0;

    // If non flat top triangle, fill flat bottom part
    if (y1 != y0) {
        for (int y = y0; y < y1; y++) {
            // (y - y1) start at -y1 comes to 0
            // inv_slop_1 is either negative or positive
            // so x_start starts at x0
            int x_start = x1 + (y - y1) * inv_slop_1;
            // (y - y0) always positive and starts at y0 and goes to y_1
            // inv_slop2 is either positive or negative
            // x_end starts at x0 too
            int x_end = x0 + (y - y0) * inv_slop_2;

            if (x_end < x_start) {
                swap(&x_end, &x_start);
            }

            for (int x = x_start; x < x_end; x++) {
                draw_texel(x, y, texture, point_a, point_b, point_c, a_uv, b_uv, c_uv);
            }
        }
    }

    inv_slop_1 = y1 != y2
        ? (float) (x2 - x1) / abs(y2 - y1)
        : 0;
    inv_slop_2 = y2 != y0
        ? (float) (x2 - x0) / abs(y2 - y0)
        : 0;
    // if not flat bottom, fill flat top part
    if (y2 != y1) {
        for (int y = y1; y <= y2; ++y) {
            // (y - y1) starts at 0 raise to (y2 - y1)
            // x_start starts at x1
            int x_start = x1 + (y - y1) * inv_slop_1;
            // (y - y0) starts at m_y raise to y2
            // x_end starts at x_m and goes to x2
            int x_end = x0 + (y - y0) * inv_slop_2;

            if (x_end < x_start) {
                swap(&x_end, &x_start);
            }

            for (int x = x_start; x < x_end; x++) {
                draw_texel(x, y, texture, point_a, point_b, point_c, a_uv, b_uv, c_uv);
            }
        }
    }
}