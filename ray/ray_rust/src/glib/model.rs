use gl;
use gltf;
use crate::gltf_models;
use std::rc::Rc;
use std::cell::RefCell;
use std::rc::Weak;
use std::collections::HashMap;


#[derive(Clone)]
pub struct Model {
    pub gtlf_model: gltf_models::Model,
    pub scenes: Vec<Scene>,
}

#[derive(Clone)]
pub struct Scene {
    pub gtlf_scene: gltf_models::Scene,
    pub nodes: Vec<Rc<RefCell<Node>>>,
}

#[derive(Clone)]
pub struct Node {
    pub gtlf_node: Rc<RefCell<gltf_models::Node>>,
    pub children: Vec<Rc<RefCell<Node>>>,
    pub parent: Option<Weak<RefCell<Node>>>,
    pub mesh: Option<Rc<Mesh>>,
}

#[derive(Clone)]
pub struct Mesh {
    pub gtlf_mesh: gltf_models::Mesh,
    pub primitives: Vec<MeshPrimitive>,
}

#[derive(Clone)]
pub struct MeshPrimitive {
    pub gtlf_primitive: gltf_models::MeshPrimitive,
    pub attributes: HashMap<gltf::Semantic, GpuAccessor>,
    pub indices: Option<GpuAccessor>,
    pub material: Option<Material>,
    pub vao: u32,
    pub vbos: Vec<u32>,
    pub ebo: Option<u32>,
}

#[derive(Clone)]
pub struct GpuAccessor {
    pub gtlf_accessor: gltf_models::GpuAccessor,
    pub buffer_view: BufferView,
}

#[derive(Clone)]
pub struct BufferView {
    pub gtlf_buffer_view: gltf_models::GpuBufferView,
    pub buffer_id: u32,
}

#[derive(Clone)]
pub struct Material {
    pub gltf_material: gltf_models::Material,
    pub pbr_metallic_roughness: PbrMetallicRoughness,
    pub normal_texture: Option<NormalTexture>,
    pub occlusion_texture: Option<OcclusionTexture>,
    pub emissive_texture: Option<EmissiveTexture>,
}

#[derive(Clone)]
pub struct PbrMetallicRoughness {
    pub gltf_pbr_metallic_roughness: gltf_models::PbrMetallicRoughness,
    pub base_color_texture: Option<TextureInfo>,
    pub metallic_roughness_texture: Option<TextureInfo>,
}

#[derive(Clone)]
pub struct NormalTexture {
    pub gltf_normal_texture: gltf_models::NormalTexture,
    pub normal_texture: TextureInfo,
}

#[derive(Clone)]
pub struct OcclusionTexture {
    pub gltf_occlusion_texture: gltf_models::OcclusionTexture,
    pub occlusion_texture: TextureInfo,
}

#[derive(Clone)]
pub struct EmissiveTexture {
    pub gltf_emissive_texture: gltf_models::EmissiveTexture,
    pub emissive_texture: TextureInfo,
}

#[derive(Clone)]
pub struct TextureInfo {
    pub gltf_texture_info: gltf_models::TextureInfo,
    pub id: u32,
    pub texture_modes: TextureModes,
}


#[derive(Clone)]
pub struct TextureModes {
    pub wrap_mode_s: gl::types::GLenum,
    pub wrap_mode_t: gl::types::GLenum,
    pub magnifying_filter: gl::types::GLenum,
    pub minifying_filter: gl::types::GLenum,
    pub border: [f32; 4],
    pub mipmaps: bool,
    pub bit_format: gl::types::GLenum,
}