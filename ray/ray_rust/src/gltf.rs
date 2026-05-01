use std::path::Path;
use std::rc::Rc;
use std::collections::HashMap;
use wgpu::util::DeviceExt;
use gltf::accessor::{DataType, Dimensions};


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
    pub indices: Option<usize>,
    pub material: usize,
    pub mode: u32,
}

pub struct GpuBufferView {
    pub buf: wgpu::Buffer,            // dedup target; Arc-shared internally
    pub offset: wgpu::BufferAddress,  // view.offset() — within source blob
    pub length: wgpu::BufferAddress,  // view.length()
    pub byte_stride: Option<u32>,     // view.stride(); None = tightly packed
}
pub struct GpuAccessor {
    pub view: GpuBufferView,
    pub offset: wgpu::BufferAddress,  // accessor.offset() — RELATIVE to view start
    pub count: usize,                   // accessor.count() — element count, not bytes
    pub component_type: DataType,     // I8/U8/I16/U16/U32/F32
    pub dimensions: Dimensions,       // Scalar/Vec2/Vec3/Vec4/Mat*
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
    let mut buffers_usages: HashMap<usize, wgpu::BufferUsages> = HashMap::new();

    gltf.document.meshes().for_each(|mesh: gltf::Mesh<'_>| {
        mesh.primitives().for_each(|primitive| {
            let Some(indices_accessor) = primitive.indices() else {
                return;
            };
            let Some(index_buffer_view) = indices_accessor.view() else {
                return;
            };
            *buffers_usages.entry(index_buffer_view.index())
                .or_insert(wgpu::BufferUsages::empty()) |= wgpu::BufferUsages::INDEX;

            primitive.attributes().for_each(|(_, attribute_accessor)| {
                let Some(buffer_view) = attribute_accessor.view() else {
                    return;
                };
                *buffers_usages.entry(buffer_view.buffer().index())
                    .or_insert(wgpu::BufferUsages::empty()) |= wgpu::BufferUsages::VERTEX;
            });
        });
    });

    let mut buffers: HashMap<usize, wgpu::Buffer> = HashMap::new();
    gltf.document.buffers().for_each(|buffer| {
        let buffer_index = buffer.index();
        if buffers.get(&buffer_index).is_some() { return; };

        let buffer_usages = buffers_usages.get(&buffer_index).unwrap();
        let buffer_data = &gltf.buffers[buffer_index];
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Buffer({})", buffer_index)),
            contents: buffer_data.0.as_slice(),
            usage: *buffer_usages,
        });

        buffers.insert(buffer_index, buffer);
    });

    let mut accessors: HashMap<usize, GpuAccessor> = HashMap::new();
    gltf.document.accessors().for_each(|accessor| {
        let Some(accessor_view) = accessor.view() else {
            panic!("Sparse accessor not supported");
        };

        let gpu_accessor = GpuAccessor {
            view: GpuBufferView {
                buf: buffers.get(&accessor_view.buffer().index()).expect("Buffer not found").clone(),
                offset: accessor_view.offset() as u64,
                length: accessor_view.length() as u64,
                byte_stride: accessor_view.stride().map(|s| s as u32),
            },
            offset: accessor.offset() as u64,
            count: accessor.count(),
            component_type: accessor.data_type(),
            dimensions: accessor.dimensions(),
        };
        accessors.insert(accessor.index(), gpu_accessor);
    });
    
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