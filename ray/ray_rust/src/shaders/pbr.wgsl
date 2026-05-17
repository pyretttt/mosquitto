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
    @builtin(position) clip_position: vec4<f32>
    @location(0) normal: vec3<f32>,
    @location(1) tangent: vec3<f32>,
    @location(2) tex_coords0: vec2<f32>,
    @location(3) tex_coords1: vec2<f32>,
    @location(4) color: vec3<f32>,
    @location(5) joint: vec3<f32>,
    @location(6) weight: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> model_mat: mat4x4<f32>
@group(0) @binding(1)
var<uniform> view_mat: mat4x4<f32>
@group(0) @binding(2)
var<uniform> proj_mat: mat4x4<f32>

@group(0)

@vertex
fn vertex_main(
    vertex: VertexInput
) -> VertexOutput {

}

// Fragment

fn fragment_main(
    in: VertexOutput
) -> @location(0) vec4<f32> {

}