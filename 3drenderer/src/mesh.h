#ifndef MESH_H
#define MESH_H

#include "vector.h"
#include "triangle.h"
#include "upng.h"

typedef struct {
    vec3_t *vertices;
    face_t *faces;
    upng_t *texture;
    vec3_t rotation;
    vec3_t scale;
    vec3_t translation;
} mesh_t;

void load_mesh_obj_data(mesh_t *mesh, char *filename);
int get_num_meshes(void);
mesh_t* get_mesh(int index);

void load_mesh(
    char *obj_filename,
    char *png_filename,
    vec3_t scale,
    vec3_t translation,
    vec3_t rotation
);
void free_meshes(void);

#endif