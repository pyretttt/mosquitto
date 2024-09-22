#include <stdio.h>
#include <stdlib.h>

#include "mesh.h"
#include "array.h"

mesh_t mesh = {
    .vertices = NULL,
    .faces = NULL,
    .rotation = {0, 0, 0},
    .scale = {1, 1, 1},
    .translation = {0, 0, 0}
};

vec3_t cube_vertices[N_CUBE_VERTICES] = {
    { .x = -1, .y = -1, .z = -1 }, // 1
    { .x = -1, .y =  1, .z = -1 }, // 2
    { .x =  1, .y =  1, .z = -1 }, // 3
    { .x =  1, .y = -1, .z = -1 }, // 4
    { .x =  1, .y =  1, .z =  1 }, // 5
    { .x =  1, .y = -1, .z =  1 }, // 6
    { .x = -1, .y =  1, .z =  1 }, // 7
    { .x = -1, .y = -1, .z =  1 }  // 8
};

face_t cube_faces[N_CUBE_FACES] = {
    // front
    { .a = 1, .b = 2, .c = 3, .color = 0xFFFFFFFF },
    { .a = 1, .b = 3, .c = 4, .color = 0xFFFFFFFF },
    // right
    { .a = 4, .b = 3, .c = 5, .color = 0xFFFFFFFF },
    { .a = 4, .b = 5, .c = 6, .color = 0xFFFFFFFF },
    // back
    { .a = 6, .b = 5, .c = 7, .color = 0xFFFFFFFF },
    { .a = 6, .b = 7, .c = 8, .color = 0xFFFFFFFF },
    // left
    { .a = 8, .b = 7, .c = 2, .color = 0xFFFFFFFF },
    { .a = 8, .b = 2, .c = 1, .color = 0xFFFFFFFF },
    // top
    { .a = 2, .b = 7, .c = 5, .color = 0xFFFFFFFF },
    { .a = 2, .b = 5, .c = 3, .color = 0xFFFFFFFF },
    // bottom
    { .a = 6, .b = 8, .c = 1, .color = 0xFFFFFFFF },
    { .a = 6, .b = 1, .c = 4, .color = 0xFFFFFFFF }
};

void load_cube_mesh_data(void) {
    for (int i = 0; i < N_CUBE_VERTICES; ++i) {
        array_push(mesh.vertices, cube_vertices[i]);
    }

    for (int i = 0; i < N_CUBE_FACES; ++i) {
        array_push(mesh.faces, cube_faces[i]);
    }
}

void load_obj_file_data(char *filename) {
    FILE *obj_file = fopen(filename, "r");
    if (!obj_file)
    {
        printf("failed to read file %s", filename);
        exit(1);
    }

    char line[256];
    while (fgets(line, 256, obj_file)) {
        vec3_t vertex;
        if (sscanf(line, "v %f %f %f", &vertex.x, &vertex.y, &vertex.z) == 3)
        {
            array_push(mesh.vertices, vertex);
        }

        face_t face;
        face.color = 0xFFFFFFFF;
        if (sscanf(line, "f %i/%*i/%*i %i/%*i/%*i %i/%*i/%*i", &face.a, &face.b, &face.c) == 3)
        {
            array_push(mesh.faces, face);
        }
    }

    fclose(obj_file);
}