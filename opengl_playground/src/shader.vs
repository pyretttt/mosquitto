#version 410 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 color;

uniform float xShift;

out vec4 vertexColor;
out vec3 vertexPos;

void main() {
    vertexColor = vec4(color, 1.0);
    gl_Position = vec4(aPos, 1.0);
    vertexPos = gl_Position.xyz;
}