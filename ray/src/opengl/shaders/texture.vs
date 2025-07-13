#version 410 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texCoord;

// Outputs
out vec3 FragWorldPos;
out vec2 TexCoord;

void main() {
    FragWorldPos = vec3(aPos);
    TexCoord = texCoord;
    
    gl_Position = vec4(FragWorldPos, 1.0);
}