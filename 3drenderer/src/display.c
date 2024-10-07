#include <math.h>

#include "display.h"

static SDL_Window* window = NULL;
static SDL_Renderer* renderer = NULL;
static uint32_t* color_buffer = NULL;
static float* z_buffer = NULL;
static SDL_Texture* color_buffer_texture = NULL;
static int window_width = 800;
static int window_height = 600;
static int cull_method;
static int render_mode;

int get_window_width(void) {
    return window_width;
}

int get_window_height(void) {
    return window_height;
}

void set_render_method(int render_method) {
    render_mode = render_method;
}
int get_render_method(void) {
    return render_mode;
}

void set_cull_method(int value) {
    cull_method = value;
}
int get_cull_method(void) {
    return cull_method;
}

SDL_Renderer* get_renderer(void) {
    return renderer;
}

uint32_t* get_color_buffer() {
    return color_buffer;
}

float* get_z_buffer() {
    return z_buffer;
}

bool initialize_window(void) {
    if (SDL_Init(SDL_INIT_EVERYTHING) != 0) {
        fprintf(stderr, "Error initializing SDL.\n");
        return false;
    }

    // Set width and height of the SDL window with the max screen resolution
    SDL_DisplayMode display_mode;
    SDL_GetCurrentDisplayMode(0, &display_mode);

    // Create a SDL Window
    window = SDL_CreateWindow(
        NULL,
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        window_width,
        window_height,
        SDL_WINDOW_BORDERLESS
    );
    if (!window) {
        fprintf(stderr, "Error creating SDL window.\n");
        return false;
    }

    // Create a SDL renderer
    renderer = SDL_CreateRenderer(window, -1, 0);
    if (!renderer) {
        fprintf(stderr, "Error creating SDL renderer.\n");
        return false;
    }

    color_buffer = (uint32_t *) malloc(sizeof(uint32_t) 
        * window_height
        * window_width);
    z_buffer = (float *) malloc(
        sizeof(float) 
        * window_height
        * window_width
    );
    color_buffer_texture = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STREAMING,
        window_width,
        window_height
    );

    return true;
}

void draw_grid(void) {
    for (int y = 0; y < window_height; y += 10) {
        for (int x = 0; x < window_width; x += 10) {
            color_buffer[(window_width * y) + x] = 0xFF444444;
        }
    }
}

void draw_pixel(int x, int y, uint32_t color) {
    if (x >= 0 && y >= 0 && x < window_width && y < window_height) {
        color_buffer[y * window_width + x] = color;
    }
}

void draw_rect(int x, int y, int width, int height, uint32_t color) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            int current_x = x + i;
            int current_y = y + j;
            draw_pixel(current_x, current_y, color);
        }
    }
}

void draw_line_s(int x0, int y0, int x1, int y1, uint32_t color) {
    int dx = x1 - x0;
    int dy = y1 - y0;
    
    int dist = abs(dx) >= abs(dy) ? abs(dx) : abs(dy);  
    float x_step = dx / (float) dist;
    float y_step = dy / (float) dist;

    int i = 0;
    while (i <= dist) {
        float x_pos = (float) x0 + x_step * i;
        float y_pos = (float) y0 + y_step * i;

        int x_cand = (int) round(x_pos);
        int y_cand = (int) round(y_pos);

        color_buffer[y_cand * window_width + x_cand] = color;
        ++i;
    }
}

void draw_line(int x0, int y0, int x1, int y1, uint32_t color) {
    int dx = x1 - x0;
    int dy = y1 - y0;
    
    int side_lenght = abs(dx) >= abs(dy) ? abs(dx) : abs(dy);  
    float x_inc = dx / (float) side_lenght;
    float y_inc = dy / (float) side_lenght;

    float current_x = x0;
    float current_y = y0;
    for (int i = 0; i <= side_lenght; ++i) {
        draw_pixel(round(current_x), round(current_y), color);
        current_x += x_inc;
        current_y += y_inc;
    }
}

void draw_triangle(int x0, int y0, int x1, int y1, int x2, int y2, uint32_t color){
    draw_line(x0, y0, x1, y1, color);
    draw_line(x0, y0, x2, y2, color);
    draw_line(x2, y2, x1, y1, color);
}

void render_color_buffer(void) {
    SDL_UpdateTexture(
        color_buffer_texture,
        NULL,
        color_buffer,
        (int)(window_width * sizeof(uint32_t))
    );
    SDL_RenderCopy(renderer, color_buffer_texture, NULL, NULL);
}

void clear_color_buffer(uint32_t color) {
    for (int i = 0; i < window_width * window_height; i++) {
        color_buffer[i] = color;
    }
}

void clear_z_buffer() {
    for (int i = 0; i < window_width * window_height; i++) {
        z_buffer[i] = 0.0f;
    }
}

void destroy_window(void) {
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}
