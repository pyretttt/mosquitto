#version 410 core

in vec2 TexCoord;
in vec3 FragWorldPos;
in vec3 Normal;

uniform sampler2D ambient0;
uniform sampler2D specular0;
uniform sampler2D diffuse0;
uniform sampler2D normal0;


uniform struct Light {
    vec3 position;
    vec3 intensity;
} light;

uniform float shiness;
uniform vec3 cameraPos;

out vec4 FragColor;

void main() {
    vec3 viewDirection = normalize(cameraPos - FragWorldPos);
    vec3 lightDirection = normalize(light.position - FragWorldPos);
    vec3 reflectedLightDirection = reflect(-lightDirection, Normal);
    float spec = pow(max(dot(viewDirection, reflectedLightDirection), 0.0), min(shiness, 0.01) * 128.0);
    
    float specularStrength = 0.5;
    vec4 specular = vec4(specularStrength * spec) * texture(specular0, TexCoord);

    float diffuseMagnitude = max(dot(lightDirection, Normal), 0);
    FragColor = (texture(diffuse0, TexCoord) * diffuseMagnitude) + specular;
}