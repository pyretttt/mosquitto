#ifndef CAMERA_H
#define CAMERA_H

#include "vector.h"
#include "matrix.h"

typedef struct {
    vec3_t position;
    vec3_t direction;
    vec3_t forward_velocity;
    float yaw_angle;
    float pitch;
} camera_t;

extern camera_t camera;

vec3_t get_camera_look_at_target(void);

#endif