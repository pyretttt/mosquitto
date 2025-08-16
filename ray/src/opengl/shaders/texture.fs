#version 410 core

in vec3 FragWorldPos;
in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D texture0;

void main() {
    FragColor = texture(texture0, TexCoord);
}