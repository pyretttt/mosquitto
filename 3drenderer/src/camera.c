#include "camera.h"
#include "matrix.h"

camera_t camera = {
    .direction = {0, 0, 0}, 
    .position = {0, 0, 1},
    .forward_velocity = {0, 0, 0},
    .yaw_angle = 0.0
};