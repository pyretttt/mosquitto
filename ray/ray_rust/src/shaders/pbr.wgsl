// Vertex

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec3<f32>,
    @location(3) tex_coords0: vec2<f32>,
    @location(4) tex_coords1: vec2<f32>,
    @location(5) color: vec3<f32>,
    @location(6) joint: vec3<f32>,
    @location(7) weight: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) tangent: vec3<f32>,
    @location(2) tex_coords0: vec2<f32>,
    @location(3) tex_coords1: vec2<f32>,
    @location(4) color: vec3<f32>,
    @location(5) joint: vec3<f32>,
    @location(6) weight: vec3<f32>,
}

struct TransformUniforms {
    model: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
}
@group(0) @binding(0)
var<uniform> transforms: TransformUniforms;

fn transform_direction(m: mat4x4<f32>, v: vec3<f32>) -> vec3<f32> {
    return normalize((m * vec4<f32>(v, 0.0)).xyz);
}

@vertex
fn vertex_main(vertex: VertexInput) -> VertexOutput {
    let world_pos = transforms.model * vec4<f32>(vertex.position, 1.0);
    let clip_pos = transforms.proj * transforms.view * world_pos;

    return VertexOutput(
        clip_pos,
        transform_direction(transforms.model, vertex.normal),
        transform_direction(transforms.model, vertex.tangent),
        vertex.tex_coords0,
        vertex.tex_coords1,
        vertex.color,
        vertex.joint,
        vertex.weight,
    );
}

// Fragment

@group(1) @binding(0)
var t_base_color: texture_2d<f32>;
@group(1) @binding(1)
var s_base_color: sampler;

struct MaterialUniforms {
    base_color_factor: vec4<f32>,
    metallic_factor: f32,
    roughness_factor: f32,
    _pad: vec2<f32>,
}
@group(1) @binding(2)
var<uniform> material: MaterialUniforms;

@fragment
fn fragment_main(frag: VertexOutput) -> @location(0) vec4<f32> {
    let base_sample = textureSample(t_base_color, s_base_color, frag.tex_coords0);
    var base_color = base_sample * material.base_color_factor;
    base_color = vec4(base_color.rgb * frag.color, base_color.a);

    let light_dir = normalize(vec3(0.35, 0.85, 0.4));
    let n = normalize(frag.normal);
    let ndotl = max(dot(n, light_dir), 0.0);
    let lit = base_color.rgb * (0.12 + 0.88 * ndotl);

    return vec4(lit, base_color.a);
}
