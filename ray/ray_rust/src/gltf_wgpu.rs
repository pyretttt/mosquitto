use crate::gltf;
use gltf;

pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
    pub tex_coords0: [f32; 2],
    pub tex_coords1: [f32; 2],
    pub color: [f32; 3],
    pub joint: [f32; 3],
    pub weight: [f32; 3],
}

pub fn map_semantic(semantic: &gltf::Semantic) -> u32 {
    match semantic {
        gltf::Semantic::Positions => 0,
        gltf::Semantic::Normals => 1,
        gltf::Semantic::Tangents => 2,
        gltf::Semantic::TexCoords(x) if *x < 2 => 3 + x,
        gltf::Semantic::Colors(x) if *x == 0 => 5,
        gltf::Semantic::Joints(x) if *x == 0 => 6,
        gltf::Semantic::Weights(x) if *x == 0 => 7,
        _ => panic!("Unsupported semantic: {:?}", semantic),
    }
}
