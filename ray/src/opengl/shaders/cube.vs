#version 410 core
layout (location = 0) in vec3 aPos;

out vec3 TexCoords;

uniform struct Transforms {
    mat4 worldMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
} transforms;


void main() {
    TexCoords = aPos;
    gl_Position = transforms.projectionMatrix * transforms.viewMatrix * vec4(aPos, 1.0);
}