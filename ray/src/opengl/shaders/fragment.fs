#version 410 core

in vec2 TexCoord;
in vec3 FragPos;
in vec3 Normal;
in vec3 LightPos;

uniform sampler2D ambient0;
uniform sampler2D specular0;
uniform sampler2D diffuse0;
uniform sampler2D normal0;

out vec4 FragColor;

void main() {
    float diffuseMagnitude = max(dot(normalize(LightPos - FragPos), Normal), 0);
    FragColor = (texture(diffuse0, TexCoord) * diffuseMagnitude);
}