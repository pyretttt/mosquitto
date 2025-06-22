#version 410 core

in vec2 TexCoord;
in vec3 FragWorldPos;
in vec3 Normal;

uniform struct Light {
    vec3 position;
    vec3 direction;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    float attenuanceConstant;
    float attenuanceLinear;
    float attenuanceQuadratic;
} light;

uniform struct Material {
    sampler2D ambient0;
    sampler2D specular0;
    sampler2D diffuse0;
    sampler2D normal0;
    
    float shiness;
} material;

uniform vec3 cameraPos;

out vec4 FragColor;

void main() {
    vec3 lightDirection = normalize(light.position - FragWorldPos);
    float distance = length(light.position - FragWorldPos);
    float attenuation = 1.0 / (
        light.attenuanceConstant + light.attenuanceLinear * distance + light.attenuanceQuadratic * distance * distance
    );

    // ambient
    vec4 ambient = vec4(light.ambient, 1.0) * texture(material.ambient0, TexCoord);

    // specular
    vec3 viewDirection = normalize(cameraPos - FragWorldPos);
    vec3 reflectedLightDirection = reflect(-lightDirection, Normal);
    float spec = pow(max(dot(viewDirection, reflectedLightDirection), 0.0), material.shiness * 128.0);
    vec4 specular = vec4(light.specular, 1.0) * spec * texture(material.specular0, TexCoord);

    // diffuse
    float diffuseMagnitude = max(dot(lightDirection, Normal), 0.0);
    vec4 diffuse = vec4(light.diffuse, 1.0) * diffuseMagnitude * texture(material.diffuse0, TexCoord);

    ambient *= attenuation;
    specular *= attenuation;
    diffuse *= attenuation;

    FragColor = ambient + diffuse + specular;
}