#version 410 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texCoord;


struct Transforms {
    mat4 worldMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
};

uniform Transforms transforms;
uniform vec3 lightPos;
uniform vec3 cameraPos;

out vec2 TexCoord;
out vec3 FragPos;
out vec3 Normal;
out vec3 LightPos;
out vec3 CameraPos;

void main() {
    vec4 worldPosition = transforms.worldMatrix * vec4(aPos, 1.0);
    FragPos = vec3(worldPosition);
    TexCoord = texCoord;
    Normal = normalize(mat3(inverse(transpose(transforms.worldMatrix))) * normal);
    LightPos = lightPos;
    CameraPos = cameraPos;
    
    gl_Position = transforms.projectionMatrix * transforms.viewMatrix * worldPosition;
}