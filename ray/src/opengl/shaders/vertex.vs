#version 410 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texCoord;

uniform mat4 transform;

struct Transforms {
    mat4 worldMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
};

uniform Transforms transforms;

out vec2 TexCoord;
out vec3 FragPosition;
out vec3 Normal;

void main() {
    vec4 worldPosition = transforms.worldMatrix * vec4(aPos, 1.0);
    FragPosition = vec3(worldPosition);
    TexCoord = texCoord;
    Normal = normal;
    
    gl_Position = transforms.projectionMatrix * transforms.viewMatrix * worldPosition;
}