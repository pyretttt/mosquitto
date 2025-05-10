#version 410 core

in vec4 vertexColor;
uniform vec4 ourColor;

out vec4 FragColor;

void main() {
    FragColor = ourColor;
}