#version 410 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec4 color;
layout (location = 2) in vec2 texCoord;

uniform mat4 transform;

out vec2 TexCoord;
out vec4 vertexColor;

void main() {
    gl_Position = transform * vec4(aPos, 1.0);
    vertexColor = color;
    TexCoord = texCoord;
}