#version 410 core
out vec4 FragColor;
in vec4 vertexColor;
in vec3 vertexPos;

void main() {
    FragColor = vec4(vertexPos, 1.0);
}