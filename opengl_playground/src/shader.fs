#version 410 core

in vec4 ourColor;
in vec2 TexCoord;

uniform sampler2D texture0;
uniform sampler2D texture1;
uniform float uMix;

out vec4 FragColor;

void main() {
    FragColor = mix(texture(texture0, TexCoord), texture(texture1, TexCoord), uMix);
}