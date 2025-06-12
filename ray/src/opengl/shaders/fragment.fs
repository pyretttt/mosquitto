#version 410 core

in vec2 TexCoord;
uniform sampler2D ambient0;
uniform sampler2D ambient1;

out vec4 FragColor;


void main() {
    FragColor = mix(
        texture(ambient0, TexCoord),
        texture(ambient1, TexCoord),
        0.8
    );
}