#version 410 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texCoord;

// Uniforms
uniform struct Transforms {
    mat4 worldMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
} transforms;

// Outputs
out vec3 FragWorldPos;

void main() {
    vec4 worldPosition = transforms.worldMatrix * vec4(aPos, 1.0);
    FragWorldPos = vec3(worldPosition);
    
    gl_Position = transforms.projectionMatrix * transforms.viewMatrix * worldPosition;
}