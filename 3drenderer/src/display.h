#ifndef DISPLAY_H
#define DISPLAY_H

#include <stdint.h>
#include <stdbool.h>

#include <SDL2/SDL.h>

#define FPS 30
#define FRAME_TARGET_TIME (1000 / FPS)

enum cull_mode {
    CULL_NONE,
    CULL_BACKFACE
};

enum rendering_method {
    RENDER_WIRE,
    RENDER_WIRE_VERTEX,
    RENDER_FILL_TRIANGLE,
    RENDER_FILL_TRIANGLE_WIRE,
    RENDER_TEXTURED,
    RENDER_TEXTURED_WIRE,
};

int get_window_width(void);
int get_window_height(void);
void set_render_method(int);
int get_render_method(void);
void set_cull_method(int);
int get_cull_method(void);
SDL_Renderer* get_renderer(void);
uint32_t* get_color_buffer();
float* get_z_buffer();

bool initialize_window(void); 
void draw_grid(void);
void draw_pixel(int x, int y, uint32_t color);
void draw_rect(int x, int y, int width, int height, uint32_t color);
void draw_line(int x0, int y0, int x1, int y1, uint32_t color);
void draw_line_s(int x0, int y0, int x1, int y1, uint32_t color);
void draw_triangle(int x0, int y0, int x1, int y1, int x2, int y2, uint32_t color);
void render_color_buffer(void); 
void clear_color_buffer(uint32_t color);
void clear_z_buffer();
void destroy_window(void);

#endif