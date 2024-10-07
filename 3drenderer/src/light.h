#ifndef LIGHT_H
#define LIGHT_H

#include <stdint.h>

#include <vector.h>

typedef struct {
    vec3_t direction;
} light_t;

void init_light(vec3_t direction);
light_t get_light(void);
uint32_t light_apply_intensity(uint32_t color, float intensity);

#endif