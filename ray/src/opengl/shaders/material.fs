#version 410 core

#define MAX_LIGHTS 16

in vec2 TexCoord;
in vec3 FragWorldPos;
in vec3 Normal;

struct Light {
    vec3 position;
    vec3 spotDirection;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    float cutoff;
    float cutoffDecay;

    float attenuanceConstant;
    float attenuanceLinear;
    float attenuanceQuadratic;
};

uniform Light light[MAX_LIGHTS];

uniform struct Material {
    sampler2D ambient0;
    sampler2D specular0;
    sampler2D diffuse0;
    sampler2D normal0;
    
    float shiness;
} material;

uniform vec3 cameraPos;
uniform int numLights;

out vec4 FragColor;


vec4 calcLightIntensity(
    Light light,
    vec4 ambientTexel,
    vec4 diffuseTexel,
    vec4 specularTexel
) {
    vec3 lightDirection = light.position - FragWorldPos;
    float distance = length(lightDirection);
    lightDirection = normalize(lightDirection);
    float attenuation = 1.0 / (
        light.attenuanceConstant + light.attenuanceLinear * distance + light.attenuanceQuadratic * distance * distance
    );

    float lightCosSim = dot(lightDirection, -light.spotDirection);
    float lightAngle = acos(lightCosSim);
    float cutoff = 1.0 - clamp((lightAngle - light.cutoff) / light.cutoffDecay, 0.0, 1.0);

    // ambient
    vec4 ambient = vec4(light.ambient, 1.0) * ambientTexel;

    // specular
    vec3 viewDirection = normalize(cameraPos - FragWorldPos);
    vec3 reflectedLightDirection = reflect(-lightDirection, Normal);
    float spec = pow(max(dot(viewDirection, reflectedLightDirection), 0.0), material.shiness * 128.0);
    vec4 specular = vec4(light.specular, 1.0) * cutoff * spec * specularTexel;

    // diffuse
    float diffuseMagnitude = max(dot(lightDirection, Normal), 0.0);
    vec4 diffuse = vec4(light.diffuse, 1.0) * cutoff * diffuseMagnitude * diffuseTexel;

    ambient *= attenuation;
    specular *= attenuation;
    diffuse *= attenuation;

    return ambient + diffuse + specular;
}

void main() {
    vec4 ambientTexel = texture(material.ambient0, TexCoord);
    vec4 specularTexel = texture(material.specular0, TexCoord);
    vec4 diffuseTexel = texture(material.diffuse0, TexCoord);

    vec4 result = vec4(0);
    for (int i = 0; i < numLights; i++) {
        result += calcLightIntensity(
            light[i],
            ambientTexel,
            diffuseTexel,
            specularTexel
            // vec4(0.2, 0.3, 0.4, 1.0),
            // vec4(0.8, 0.3, 0.3, 1.0),
            // vec4(0.2, 0.6, 0.1, 1.0)
        );
    }

    FragColor = clamp(result, 0.0, 1.0);
}