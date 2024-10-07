#include "camera.h"
#include "matrix.h"

camera_t camera = {
    .direction = {0, 0, 0}, 
    .position = {0, 0, 1},
    .forward_velocity = {0, 0, 0},
    .yaw_angle = 0.0,
    .pitch = 0.0
};

vec3_t get_camera_look_at_target(void) {
    vec3_t target = {0, 0, 1};
    mat4_t camera_yaw_rotation = make_rotation_matrix_y(camera.yaw_angle);
    mat4_t camera_pitch_rotation = make_rotation_matrix_x(camera.pitch);

    mat4_t camera_rotation = mat4_idenity();
    camera_rotation = mat4_mul(camera_pitch_rotation, camera_rotation);
    camera_rotation = mat4_mul(camera_yaw_rotation, camera_rotation);

    camera.direction = vec3_from_vec4(
        mul_vec4(camera_rotation, vec4_from_vec3(target))
    );
    target = vec3_add(camera.position, camera.direction);
    return target;
}