use std::path::Path;
use std::rc::Rc;
use std::collections::HashMap;

#[derive(Debug)]
pub struct GLTF {
    pub document: gltf::Document,
    pub buffers: Vec<gltf::buffer::Data>,
    pub textures: Vec<gltf::image::Data>,
}

pub struct Scene {
    pub nodes: Vec<Rc<Node>>,
}

pub struct Node {
    pub children: Vec<Rc<Node>>,
    pub transform: cgmath::Matrix4<f32>,
    pub camera: Option<Rc<Camera>>,
    pub mesh: Option<Rc<Mesh>>,
}

pub enum Camera {
    Perspective(PerspectiveCamera),
    Orthographic(OrthographicCamera),
}

pub struct PerspectiveCamera {
    pub fovy: f32,
    pub aspect: f32,
    pub znear: f32,
    pub zfar: f32,
}

pub struct OrthographicCamera {
    pub xmag: f32,
    pub ymag: f32,
    pub znear: f32,
    pub zfar: f32,
}

pub struct Mesh {
    pub primitives: Vec<MeshPrimitive>,
}

pub struct MeshPrimitive {
    pub attributes: HashMap<String, usize>,
    pub indices: usize,
    pub material: usize,
    pub mode: u32,
}

pub struct BufferView {
    pub buffer: Rc<gltf::buffer::Data>,
    pub byte_offset: usize,
    pub byte_length: usize,
    pub byte_stride: usize,
    pub target: usize,
}

pub struct Accessor {
    pub buffer_view: Rc<BufferView>,
    pub byte_offset: usize,
    pub data_type: gltf::json::accessor::Type,
    pub component_type: gltf::json::accessor::ComponentType,
    pub count: usize,
    pub max: Option<Vec<f32>>,
    pub min: Option<Vec<f32>>,
}

pub fn load_gltf(path: &Path) -> Result<GLTF, gltf::Error> {
    let (document, buffers, textures) = gltf::import(path)?;
    document.meshes().map(|x| x.primitives().map(|x: gltf::Primitive<'_>| x.indices))
    Ok(GLTF { document, buffers, textures })
}