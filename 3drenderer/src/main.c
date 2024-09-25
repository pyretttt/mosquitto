#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#include "SDL.h"
#include "upng.h"

#include "display.h"
#include "vector.h"
#include "mesh.h"
#include "triangle.h"
#include "array.h"
#include "matrix.h"
#include "light.h"
#include "texture.h"

triangle_t* triangles_to_render = NULL;
bool is_running = false;
int previous_frame_time = 0;

vec3_t camera_position = {.x = 0, .y = 0, .z = 0};
mat4_t proj_mat;
float fov_factor = 640.f;
float rotation = 0.01;


void setup(void) {
    render_method = RENDER_WIRE;
    cull_mode = CULL_BACKFACE;
    color_buffer = (uint32_t *) malloc(sizeof(uint32_t) 
        * window_height
        * window_width);
    color_buffer_texture = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STREAMING,
        window_width,
        window_height
    );
    proj_mat = make_perspective_matrix(
        M_PI / 3.0f,
        (float) window_width/ window_height,
        0.1,
        100.0
    );
    
    load_obj_file_data("assets/cube.obj");
    // load_cube_mesh_data();

    load_png_texture_data("assets/cube.png");
}

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
            if (event.key.keysym.sym == SDLK_5)
                render_method = RENDER_TEXTURED;  
            if (event.key.keysym.sym == SDLK_6)
                render_method = RENDER_TEXTURED_WIRE;  
            if (event.key.keysym.sym == SDLK_c)
                cull_mode = CULL_BACKFACE;
            if (event.key.keysym.sym == SDLK_d)
                cull_mode = CULL_NONE;
            break;
    }
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
 
    mesh.translation.x += 0.001;
    mesh.translation.y += 0.001;
    mesh.translation.z = 8;

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
            transformed_vertices[j] = transformed_vertex;
        }

        vec3_t normal = cross_product(
            vec3_sub(vec3_from_vec4(transformed_vertices[1]), vec3_from_vec4(transformed_vertices[0])),
            vec3_sub(vec3_from_vec4(transformed_vertices[2]), vec3_from_vec4(transformed_vertices[0]))
        );
        vec3_normalize(&normal);
        // Backface culling
        if (cull_mode == CULL_BACKFACE) {
            
            vec3_t camera_ray = vec3_sub(camera_position, vec3_from_vec4(transformed_vertices[0]));
            float dot_product = vec3_dot_product(normal, camera_ray);

            if (dot_product < 0)
                continue;
        }

        vec4_t projected_points[3];
        for (uint j = 0; j < (sizeof(face_vertices) / sizeof(vec3_t)); ++j) {
            vec4_t perspective_projected_vertex = mat4_mul_vec4_project(proj_mat, transformed_vertices[j]);
            projected_points[j] = perspective_projected_vertex;

            projected_points[j].x *= (window_width / 2.0f);
            projected_points[j].y *= -(window_height / 2.0f);

            projected_points[j].x += (window_width / 2.0f);
            projected_points[j].y += (window_height / 2.0f);
        }
        float avg_depth = (transformed_vertices[0].z + transformed_vertices[1].z + transformed_vertices[2].z) / 3.0;

        uint32_t face_color = light_apply_intensity(
            face.color, 
            fmax(0.1, vec3_dot_product(normal, vec3_mul(light.direction, -1)))
        );

        triangle_t projected_triangle = {
            .points = {
                {projected_points[0].x, projected_points[0].y, projected_points[0].z, projected_points[0].w},
                {projected_points[1].x, projected_points[1].y, projected_points[1].z, projected_points[1].w},
                {projected_points[2].x, projected_points[2].y, projected_points[2].z, projected_points[2].w},
            },
            .tex_coords = {
                {face.a_uv.u, face.a_uv.v},
                {face.b_uv.u, face.b_uv.v},
                {face.c_uv.u, face.c_uv.v},
            },
            .color = face_color,
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

        if (render_method == RENDER_TEXTURED || render_method == RENDER_TEXTURED_WIRE) {
            draw_textured_triangle(
                triangle.points[0].x,
                triangle.points[0].y,
                triangle.points[0].z,
                triangle.points[0].w,
                triangle.tex_coords[0].u,
                triangle.tex_coords[0].v,
                triangle.points[1].x,
                triangle.points[1].y,
                triangle.points[1].z,
                triangle.points[1].w,
                triangle.tex_coords[1].u,
                triangle.tex_coords[1].v,
                triangle.points[2].x,
                triangle.points[2].y,
                triangle.points[2].z,
                triangle.points[2].w,
                triangle.tex_coords[2].u,
                triangle.tex_coords[2].v,
                mesh_texture
            );
        }

        if (render_method == RENDER_WIRE 
            || render_method == RENDER_FILL_TRIANGLE_WIRE
            || render_method == RENDER_WIRE_VERTEX
            || render_method == RENDER_TEXTURED_WIRE
        ) {
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
    free(png_texture);
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