#version 410 core

in vec4 vertexColor;
in vec2 TexCoord;
uniform sampler2D texture0;
uniform sampler2D texture1;

out vec4 FragColor;


void main() {
    FragColor = mix(
        texture(texture0, TexCoord),
        texture(texture1, TexCoord),
        0.8
    );
}