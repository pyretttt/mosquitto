#include "light.h"

static light_t light;

void init_light(vec3_t direction) {
    light = (light_t){.direction = direction};
}
light_t get_light(void) {
    return light;
}

uint32_t light_apply_intensity(uint32_t color, float intensity) {
    if (intensity < 0) intensity = 0;
    if (intensity > 1) intensity = 1;

    uint32_t a = (color & 0xFF000000);
    uint32_t r = (color & 0x00FF0000) * intensity;
    uint32_t g = (color & 0x0000FF00) * intensity;
    uint32_t b = (color & 0x000000FF) * intensity;

    uint32_t new_color = a 
        | (r & 0x00FF0000) 
        | (g & 0x0000FF00) 
        | (b & 0x000000FF);

    return new_color;
}