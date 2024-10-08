#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#include "SDL.h"
#include "upng.h"

#include "camera.h"
#include "display.h"
#include "vector.h"
#include "mesh.h"
#include "triangle.h"
#include "array.h"
#include "matrix.h"
#include "light.h"
#include "texture.h"
#include "clipping.h"

#define MAX_TRIANGLES_PER_MESH 10000
triangle_t triangles_to_render[MAX_TRIANGLES_PER_MESH];
int num_triangles_to_render = 0;

bool is_running = false;
int previous_frame_time = 0;

mat4_t proj_mat;
mat4_t view_matrix;
float fov_factor = 640.f;
float rotation = 0.1;
float delta_time = 0;

void setup(void) {
    set_render_method(RENDER_WIRE);
    set_cull_method(CULL_BACKFACE);
    init_light((vec3_t){0, 0, 1});

    float aspect_ratio = (float) get_window_width() / get_window_height();
    float fov = M_PI / 3.0f;
    float z_near = 0.1f;
    float z_far = 100.0f;
    proj_mat = make_perspective_matrix(
        fov,
        aspect_ratio,
        z_near,
        z_far
    );
    init_frustum_planes(fov, z_near, z_far, aspect_ratio);
    load_mesh(
        "./assets/crab.obj",
         "./assets/crab.png", 
        (vec3_t) {1, 1, 1}, 
        (vec3_t) {3, 0, 7}, 
        (vec3_t) {0, 0, 0}
    );
    load_mesh(
        "./assets/f22.obj",
         "./assets/f22.png", 
        (vec3_t) {1, 1, 1}, 
        (vec3_t) {-3, 0, 7}, 
        (vec3_t) {0, 0, 0}
    );
}

void process_input(void) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type)
        {
            case SDL_QUIT:
                is_running = false;
                break;
            case SDL_KEYDOWN:
                if (event.key.keysym.sym == SDLK_ESCAPE)
                    is_running = false;
                if (event.key.keysym.sym == SDLK_SPACE)
                    rotation = rotation > 0 ? 0 : 0.1;
                if (event.key.keysym.sym == SDLK_1)
                    set_render_method(RENDER_WIRE_VERTEX);
                if (event.key.keysym.sym == SDLK_2)
                    set_render_method(RENDER_WIRE);
                if (event.key.keysym.sym == SDLK_3)
                    set_render_method(RENDER_FILL_TRIANGLE);
                if (event.key.keysym.sym == SDLK_4)
                    set_render_method(RENDER_FILL_TRIANGLE_WIRE);  
                if (event.key.keysym.sym == SDLK_5)
                    set_render_method(RENDER_TEXTURED);
                if (event.key.keysym.sym == SDLK_6)
                    set_render_method(RENDER_TEXTURED_WIRE);  
                if (event.key.keysym.sym == SDLK_c)
                    set_cull_method(CULL_BACKFACE);
                if (event.key.keysym.sym == SDLK_x)
                    set_cull_method(CULL_NONE);
                if (event.key.keysym.sym == SDLK_w) {
                    camera.pitch += 3.0 * delta_time;
                }
                if (event.key.keysym.sym == SDLK_s) {
                    camera.pitch += -3.0 * delta_time;
                }
                if (event.key.keysym.sym == SDLK_RIGHT)
                    camera.yaw_angle += 1.0 * delta_time;
                if (event.key.keysym.sym == SDLK_LEFT)
                    camera.yaw_angle -= 1.0 * delta_time;
                if (event.key.keysym.sym == SDLK_UP) {
                    camera.forward_velocity = vec3_mul(camera.direction, 5.0 * delta_time);
                    camera.position = vec3_add(camera.position, camera.forward_velocity);
                }
                if (event.key.keysym.sym == SDLK_DOWN) {
                    camera.forward_velocity = vec3_mul(camera.direction, 5.0 * delta_time);
                    camera.position = vec3_sub(camera.position, camera.forward_velocity);
                }
                break;
        }
    }
}

void process_graphics_pipeline_stages(mesh_t *m) {
    mesh_t mesh = *m;
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
            mesh.vertices[face.a],
            mesh.vertices[face.b],
            mesh.vertices[face.c]
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
            transformed_vertex = mul_vec4(view_matrix, transformed_vertex);
            transformed_vertices[j] = transformed_vertex;
        }

        vec3_t normal = cross_product(
            vec3_sub(vec3_from_vec4(transformed_vertices[1]), vec3_from_vec4(transformed_vertices[0])),
            vec3_sub(vec3_from_vec4(transformed_vertices[2]), vec3_from_vec4(transformed_vertices[0]))
        );
        vec3_normalize(&normal);

        // Backface culling
        if (get_cull_method() == CULL_BACKFACE) {
            vec3_t origin = {0, 0, 0};
            vec3_t camera_ray = vec3_sub(origin, vec3_from_vec4(transformed_vertices[0]));
            float dot_product = vec3_dot_product(normal, camera_ray);

            if (dot_product < 0)
                continue;
        }

        // Clipping before perspective projection
        polygon_t polygon = create_polygon_from_triangles(
            vec3_from_vec4(transformed_vertices[0]),
            vec3_from_vec4(transformed_vertices[1]),
            vec3_from_vec4(transformed_vertices[2]),
            face.a_uv,
            face.b_uv,
            face.c_uv
        );
        clip_polygon(&polygon);
        triangle_t triangles_after_clipping[MAX_NUM_POLY_TRIANGLES];
        int num_triangles_after_clipping = 0;
        triangles_from_polygon(&polygon, triangles_after_clipping, &num_triangles_after_clipping);

        for (int t = 0; t < num_triangles_after_clipping; t++) {
            triangle_t triangle_after_clipping = triangles_after_clipping[t];

            vec4_t projected_points[3];
            for (uint j = 0; j < (sizeof(face_vertices) / sizeof(vec3_t)); ++j) {
                vec4_t perspective_projected_vertex = mat4_mul_vec4_project(proj_mat, triangle_after_clipping.points[j]);
                projected_points[j] = perspective_projected_vertex;

                projected_points[j].x *= (get_window_width() / 2.0f);
                projected_points[j].y *= -(get_window_height() / 2.0f);

                projected_points[j].x += (get_window_width() / 2.0f);
                projected_points[j].y += (get_window_height() / 2.0f);
            }

            uint32_t face_color = light_apply_intensity(
                face.color, 
                fmax(0.1, vec3_dot_product(normal, vec3_mul(get_light().direction, -1)))
            );

            triangle_t triangle_to_render = {
                .points = {
                    {projected_points[0].x, projected_points[0].y, projected_points[0].z, projected_points[0].w},
                    {projected_points[1].x, projected_points[1].y, projected_points[1].z, projected_points[1].w},
                    {projected_points[2].x, projected_points[2].y, projected_points[2].z, projected_points[2].w},
                },
                .tex_coords = {
                    {triangle_after_clipping.tex_coords[0].u, triangle_after_clipping.tex_coords[0].v},
                    {triangle_after_clipping.tex_coords[1].u, triangle_after_clipping.tex_coords[1].v},
                    {triangle_after_clipping.tex_coords[2].u, triangle_after_clipping.tex_coords[2].v},
                },
                .color = face_color,
                .texture = mesh.texture
            };

            if (num_triangles_to_render < MAX_TRIANGLES_PER_MESH) {
                triangles_to_render[num_triangles_to_render++] = triangle_to_render;
            }
        }
    }
}

void update() {
    int time_to_delay = FRAME_TARGET_TIME - (SDL_GetTicks() - previous_frame_time);
    if (time_to_delay > 0 && time_to_delay <= FRAME_TARGET_TIME)
        SDL_Delay(time_to_delay);

    delta_time = (SDL_GetTicks() - previous_frame_time) / 1000.0f;

    previous_frame_time = SDL_GetTicks();
    num_triangles_to_render = 0;

    vec3_t up_direction = {0, 1, 0};
    vec3_t target = get_camera_look_at_target();
    view_matrix = mat4_look_at(camera.position, target, up_direction);

    for (int mesh_index = 0; mesh_index < get_num_meshes(); mesh_index++) {
        mesh_t *mesh = get_mesh(mesh_index);

        process_graphics_pipeline_stages(mesh);
    }
}

void render(void) {
    int n_triangles = num_triangles_to_render;
    for (int i = 0; i < n_triangles; ++i) {
        triangle_t triangle = triangles_to_render[i];
        
        int render_method = get_render_method();
        if (render_method == RENDER_FILL_TRIANGLE 
            || render_method == RENDER_FILL_TRIANGLE_WIRE) {
            draw_filled_triangle(
                triangle.points[0].x,
                triangle.points[0].y,
                triangle.points[0].z,
                triangle.points[0].w,
                triangle.points[1].x,
                triangle.points[1].y,
                triangle.points[1].z,
                triangle.points[1].w,
                triangle.points[2].x,
                triangle.points[2].y,
                triangle.points[2].z,
                triangle.points[2].w,
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
                triangle.texture
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

    render_color_buffer();
    clear_color_buffer(0xFF000000);
    clear_z_buffer();
    
    SDL_RenderPresent(get_renderer());
}

void free_resources(void) {
    free(get_z_buffer());
    free(get_color_buffer());
    free_meshes();
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