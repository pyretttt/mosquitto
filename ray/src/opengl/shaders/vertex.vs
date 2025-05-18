#version 410 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 texCoord;

uniform mat4 transform;

struct Transforms {
    mat4 worldMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
};

uniform Transforms transforms;

out vec2 TexCoord;

void main() {
    gl_Position = transforms.projectionMatrix * transforms.viewMatrix * transforms.worldMatrix * vec4(aPos, 1.0);
    TexCoord = texCoord;
}