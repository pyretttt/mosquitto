use std::path::Path;
use std::rc::Rc;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

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
    pub indices: Option<Rc<>,
    pub material: usize,
    pub mode: u32,
}

pub struct BufferView {
    pub buffer: Rc<wgpu::Buffer>,
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

pub struct TextureInfo {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

pub fn load_gltf(path: &Path) -> Result<GLTF, gltf::Error> {
    let (document, buffers, textures) = gltf::import(path)?;
    Ok(GLTF { document, buffers, textures })
}

pub fn make_wgpu_scenes(gltf: &GLTF, device: &wgpu::Device) -> Result<Vec<Scene>, wgpu::Error> {
    let mut buffers = Vec::<wgpu::Buffer>::new();
    buffers.reserve(gltf.buffers.len());

    let mut textures = Vec::<TextureInfo>::new();
    textures.reserve(gltf.textures.len());

    for mesh in gltf.document.meshes() {
        mesh.primitives().map(|primitive| {
            let mode = map_gltf_mesh_mode(&primitive.mode());
            // let indices =

        }.collect::<Vec<_>>();
    }


    Ok(vec![])
}

fn map_gltf_mesh_mode(mode: &gltf::mesh::Mode) -> wgpu::PrimitiveTopology {
    match mode {
        gltf::mesh::Mode::Points => wgpu::PrimitiveTopology::PointList,
        gltf::mesh::Mode::Lines => wgpu::PrimitiveTopology::LineList,
        gltf::mesh::Mode::LineStrip => wgpu::PrimitiveTopology::LineStrip,
        gltf::mesh::Mode::Triangles => wgpu::PrimitiveTopology::TriangleList,
        gltf::mesh::Mode::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
        _ => panic!("Unsupported primitive mode: {:?}", mode),
    }
}