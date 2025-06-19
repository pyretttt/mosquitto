#version 410 core

in vec2 TexCoord;
in vec3 FragPos;
in vec3 Normal;
in vec3 LightPos;
in vec3 WorldPos;

uniform sampler2D ambient0;
uniform sampler2D specular0;
uniform sampler2D diffuse0;
uniform sampler2D normal0;

out vec4 FragColor;

void main() {
    vec3 viewDirection = normalize(WorldPos - FragPos);
    float specularStrength = 0.5;
    vec3 lightDirection = normalize(LightPos - FragPos);
    vec3 reflectedLightDirection = reflect(-lightDirection, Normal);
    float spec = pow(max(dot(viewDirection, reflectedLightDirection), 0.0), 64.0);
    vec4 specular = vec4(specularStrength * spec);
    // * texture(specular0, TexCoord);

    float diffuseMagnitude = max(dot(lightDirection, Normal), 0);
    // FragColor = (texture(diffuse0, TexCoord) * diffuseMagnitude) + specular;
    FragColor = specular;
}