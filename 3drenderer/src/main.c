#include <stdio.h>
#include <stdint.h>
#include <SDL.h>
#include <stdbool.h>

#include "display.h"
#include "vector.h"
#include "mesh.h"
#include "triangle.h"
#include "array.h"
#include "matrix.h"

triangle_t* triangles_to_render = NULL;

bool is_running = false;
int previous_frame_time = 0;

vec3_t camera_position = {.x = 0, .y = 0, .z = 0};

float fov_factor = 640.f;

void setup(void) {
    render_method = RENDER_WIRE;
    cull_mode = CULL_BACKFACE;
    color_buffer = (uint32_t *) malloc(sizeof(uint32_t) 
        * window_height
        * window_width);
    color_buffer_texture = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING,
        window_width,
        window_height
    );

    // load_obj_file_data("assets/cube.obj");
    load_cube_mesh_data();
}

float rotation = 0.01;

void process_input(void) {
    SDL_Event event;
    SDL_PollEvent(&event);

    switch (event.type)
    {
        case SDL_QUIT:
            is_running = false;
            break;
        case SDL_KEYDOWN:
            if (event.key.keysym.sym == SDLK_ESCAPE)
                is_running = false;
            if (event.key.keysym.sym == SDLK_SPACE)
                rotation = rotation > 0 ? 0 : 0.01;
            if (event.key.keysym.sym == SDLK_1)
                render_method = RENDER_WIRE_VERTEX;
            if (event.key.keysym.sym == SDLK_2)
                render_method = RENDER_WIRE;
            if (event.key.keysym.sym == SDLK_3)
                render_method = RENDER_FILL_TRIANGLE;
            if (event.key.keysym.sym == SDLK_4)
                render_method = RENDER_FILL_TRIANGLE_WIRE;  
            if (event.key.keysym.sym == SDLK_c)
                cull_mode = CULL_BACKFACE;
            if (event.key.keysym.sym == SDLK_d)
                cull_mode = CULL_NONE;
            break;
    }
}

vec2_t project(vec3_t point)  
{
    vec2_t projected_point = {
        .x = fov_factor * point.x / point.z,
        .y = fov_factor * point.y / point.z
    };
    return projected_point;
}


void update() {
    int time_to_delay = FRAME_TARGET_TIME - (SDL_GetTicks() - previous_frame_time);
    if (time_to_delay > 0 && time_to_delay <= FRAME_TARGET_TIME)
        SDL_Delay(time_to_delay);

    triangles_to_render = NULL;

    previous_frame_time = SDL_GetTicks();
    mesh.rotation.y += rotation;
    mesh.rotation.z += rotation;
    mesh.rotation.x += rotation;

    mesh.scale.x += 0.002;
    mesh.scale.y += 0.002;

    mesh.translation.x += 0.01;
    mesh.translation.y += 0.01;
    mesh.translation.z = 5;

    mat4_t scale_matrix = make_scale_matrix(mesh.scale.x, mesh.scale.y, mesh.scale.z);
    mat4_t translation_matrix = make_translation_matrix(
        mesh.translation.x, 
        mesh.translation.y, 
        mesh.translation.z 
    );
    mat4_t rotation_matrix_x = make_rotation_matrix_x(mesh.rotation.x);
    mat4_t rotation_matrix_y = make_rotation_matrix_y(mesh.rotation.y);
    mat4_t rotation_matrix_z = make_rotation_matrix_z(mesh.rotation.z);
    int num_faces = array_length(mesh.faces);
    for (int i = 0; i < num_faces; ++i)
    {
        face_t face = mesh.faces[i];
        vec3_t face_vertices[3] = {
            mesh.vertices[face.a - 1],
            mesh.vertices[face.b - 1],
            mesh.vertices[face.c - 1]
        };

        vec4_t transformed_vertices[3];
        for (uint j = 0; j < (sizeof(face_vertices) / sizeof(vec3_t)); ++j)
        {
            vec4_t transformed_vertex = vec4_from_vec3(face_vertices[j]);
            
            mat4_t world_matrix = mat4_idenity();
            world_matrix = mat4_mul(scale_matrix, world_matrix);
            world_matrix = mat4_mul(rotation_matrix_x, world_matrix);
            world_matrix = mat4_mul(rotation_matrix_y, world_matrix);
            world_matrix = mat4_mul(rotation_matrix_z, world_matrix);
            world_matrix = mat4_mul(translation_matrix, world_matrix);

            transformed_vertex = mul_vec4(world_matrix, transformed_vertex);

            // vec3_t transform_vertex = vec3_rotate_x(face_vertices[j], mesh.rotation.x);
            // transform_vertex = vec3_rotate_y(transform_vertex, mesh.rotation.y);
            // transform_vertex = vec3_rotate_z(transform_vertex, mesh.rotation.z);

            transformed_vertices[j] = transformed_vertex;
        }

        // Backface culling
        if (cull_mode == CULL_BACKFACE) {
            vec3_t normal = cross_product(
                vec3_sub(vec3_from_vec4(transformed_vertices[1]), vec3_from_vec4(transformed_vertices[0])),
                vec3_sub(vec3_from_vec4(transformed_vertices[2]), vec3_from_vec4(transformed_vertices[0]))
            );
            vec3_normalize(&normal);
            
            vec3_t camera_ray = vec3_sub(camera_position, vec3_from_vec4(transformed_vertices[0]));
            float dot_product = vec3_dot_product(normal, camera_ray);

            if (dot_product < 0)
                continue;
        }

        vec2_t projected_points[3];
        for (uint j = 0; j < (sizeof(face_vertices) / sizeof(vec3_t)); ++j) {
            projected_points[j] = project(vec3_from_vec4(transformed_vertices[j]));

            projected_points[j].x += (window_width / 2);
            projected_points[j].y += (window_height / 2);
        }
        float avg_depth = (transformed_vertices[0].z + transformed_vertices[1].z + transformed_vertices[2].z) / 3.0;

        triangle_t projected_triangle = {
            .points = {projected_points[0], projected_points[1], projected_points[2]},
            .color = face.color,
            .avg_depth = avg_depth
        };

        array_push(triangles_to_render, projected_triangle);
    }

    int num_triangles = array_length(triangles_to_render);
    for (int i = 0; i < num_triangles; i++) {
        for (int j = i; j < num_triangles; j++) {
            if (triangles_to_render[i].avg_depth < triangles_to_render[j].avg_depth) {
                // Swap the triangles positions in the array
                triangle_t temp = triangles_to_render[i];
                triangles_to_render[i] = triangles_to_render[j];
                triangles_to_render[j] = temp;
            }
        }
    }
}

void render(void) {
    int n_triangles = array_length(triangles_to_render);
    for (int i = 0; i < n_triangles; ++i) {
        triangle_t triangle = triangles_to_render[i];

        if (render_method == RENDER_FILL_TRIANGLE 
            || render_method == RENDER_FILL_TRIANGLE_WIRE) {
            draw_filled_triangle(
                triangle.points[0].x,
                triangle.points[0].y,
                triangle.points[1].x,
                triangle.points[1].y,
                triangle.points[2].x,
                triangle.points[2].y,
                triangle.color
            );
        }

        if (render_method == RENDER_WIRE 
            || render_method == RENDER_FILL_TRIANGLE_WIRE
            || render_method == RENDER_WIRE_VERTEX) {
            draw_triangle(
                triangle.points[0].x,
                triangle.points[0].y,
                triangle.points[1].x,
                triangle.points[1].y,
                triangle.points[2].x,
                triangle.points[2].y,
                0xFFFF0000
            );
        }

        if (render_method == RENDER_WIRE_VERTEX) {
            draw_rect(triangle.points[0].x, triangle.points[0].y, 3, 3, 0xFFFFFF00);
            draw_rect(triangle.points[1].x, triangle.points[1].y, 3, 3, 0xFFFFFF00);
            draw_rect(triangle.points[2].x, triangle.points[2].y, 3, 3, 0xFFFFFF00);
        }
    }

    array_free(triangles_to_render);

    render_color_buffer();
    clear_color_buffer(0xFF000000);
    
    SDL_RenderPresent(renderer);
}

void free_resources(void) {
    free(color_buffer);
    array_free(mesh.vertices);
    array_free(mesh.faces);
}

int main(int argc, char *argv[]) {
    is_running = initialize_window();

    setup();

    while (is_running)
    {
        process_input();
        update();
        render();
    }

    destroy_window();
    free_resources();

    return 0;
}

// Reasons:
// 1. Backface culling