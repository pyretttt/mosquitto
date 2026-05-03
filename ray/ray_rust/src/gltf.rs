use std::path::Path;
use std::rc::Rc;
use std::rc::Weak;
use std::cell::RefCell;
use std::collections::HashMap;
use wgpu::util::DeviceExt;
use gltf::accessor::{DataType, Dimensions};
use cgmath::Matrix;


#[derive(Debug)]
pub struct GLTF {
    pub document: gltf::Document,
    pub buffers: Vec<gltf::buffer::Data>,
    pub textures: Vec<gltf::image::Data>,
}

#[derive(Clone)]
pub struct Scene {
    pub nodes: Vec<Rc<RefCell<Node>>>,
}

#[derive(Clone)]
pub struct Node {
    pub children: Vec<Rc<RefCell<Node>>>,
    pub transform: cgmath::Matrix4<f32>,
    pub camera: Option<Rc<Camera>>,
    pub mesh: Option<Rc<Mesh>>,
    pub parent: Option<Weak<RefCell<Node>>>,
}

#[derive(Clone)]
pub enum Camera {
    Perspective(PerspectiveCamera),
    Orthographic(OrthographicCamera),
}

#[derive(Clone)]
pub struct PerspectiveCamera {
    pub fovy: f32,
    pub aspect: Option<f32>,
    pub znear: f32,
    pub zfar: f32,
}

#[derive(Clone)]
pub struct OrthographicCamera {
    pub xmag: f32,
    pub ymag: f32,
    pub znear: f32,
    pub zfar: f32,
}

#[derive(Clone)]
pub struct Mesh {
    pub primitives: Vec<MeshPrimitive>,
}

#[derive(Clone)]
pub struct MeshPrimitive {
    pub attributes: HashMap<gltf::Semantic, GpuAccessor>,
    pub indices: Option<GpuAccessor>,
    pub material: Option<usize>,
    pub mode: gltf::mesh::Mode,
}

#[derive(Clone)]
pub struct GpuBufferView {
    pub buf: wgpu::Buffer,            // dedup target; Arc-shared internally
    pub offset: wgpu::BufferAddress,  // view.offset() — within source blob
    pub length: wgpu::BufferAddress,  // view.length()
    pub byte_stride: Option<u32>,     // view.stride(); None = tightly packed
}

#[derive(Clone)]
pub struct GpuAccessor {
    pub view: GpuBufferView,
    pub offset: wgpu::BufferAddress,  // accessor.offset() — RELATIVE to view start
    pub count: usize,                   // accessor.count() — element count, not bytes
    pub component_type: DataType,     // I8/U8/I16/U16/U32/F32
    pub dimensions: Dimensions,       // Scalar/Vec2/Vec3/Vec4/Mat*
}


#[derive(Clone)]
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

    let mut meshes: Vec<Rc<Mesh>> = gltf.document.meshes().map(|mesh| {
        let primitives = mesh.primitives().map(|primitive| {
            MeshPrimitive {
                attributes: primitive.attributes().map(|(semantic, attribute_accessor)| {
                    (semantic, accessors.get(&attribute_accessor.index()).expect("Accessor not found").clone())
                }).collect::<HashMap<_, _>>(),
                indices: primitive.indices().map(|indices_accessor|
                    accessors.get(&indices_accessor.index()).expect("Accessor not found").clone()
                ),
                material: primitive.material().index(),
                mode: primitive.mode(),
            }
        }).collect::<Vec<_>>();
        Rc::new(Mesh {
            primitives: primitives,
        })
    }).collect::<Vec<_>>();


    let nodes = gltf.document.nodes().map(|node| {
        Rc::new(RefCell::new(Node {
            children: vec![],
            transform: match node.transform() {
                gltf::scene::Transform::Matrix { matrix } => cgmath::Matrix4::from_cols(
                    cgmath::Vector4::from(matrix[0]),
                    cgmath::Vector4::from(matrix[1]),
                    cgmath::Vector4::from(matrix[2]),
                    cgmath::Vector4::from(matrix[3]),
                ).transpose(),
                gltf::scene::Transform::Decomposed { translation, rotation, scale } => {
                    let q = cgmath::Quaternion::new(rotation[3], rotation[0], rotation[1], rotation[2]);
                    cgmath::Matrix4::from_translation(cgmath::Vector3::from(translation))
                        * cgmath::Matrix4::from(q)
                        * cgmath::Matrix4::from_nonuniform_scale(scale[0], scale[1], scale[2])
                },
            },
            camera: match node.camera() {
                Some(camera) => {
                    match camera.projection() {
                        gltf::camera::Projection::Perspective(perspective) => {
                            Some(Rc::new(Camera::Perspective(PerspectiveCamera {
                                fovy: perspective.yfov(),
                                aspect: perspective.aspect_ratio(),
                                znear: perspective.znear(),
                                zfar: perspective.zfar().unwrap_or(f32::INFINITY),
                            })))
                        }
                        gltf::camera::Projection::Orthographic(orthographic) => {
                            Some(Rc::new(Camera::Orthographic(OrthographicCamera {
                                xmag: orthographic.xmag(),
                                ymag: orthographic.ymag(),
                                znear: orthographic.znear(),
                                zfar: orthographic.zfar(),
                            })))
                        }
                    }
                },
                None => None,
            },
            mesh: match node.mesh() {
                Some(mesh) => Some(Rc::clone(meshes.get(mesh.index()).expect("Mesh not found"))),
                None => None,
            },
            parent: None
        }))
    }).collect::<Vec<_>>();

    gltf.document.nodes().enumerate().for_each(|(index, node)| {
        let parent = nodes.get(index).expect("Node not found");

        node.children().for_each(|child| {
            let child_node = nodes.get(child.index()).expect("Node not found");
            child_node.borrow_mut().parent = Some(Rc::downgrade(parent));
            parent.borrow_mut().children.push(Rc::clone(child_node));
        })
    });

    let scenes = gltf.document.scenes().map(|scene| {
        Scene {
            nodes: scene.nodes().map(|node| {
                Rc::clone(nodes.get(node.index()).expect("Node not found"))
            }).collect::<Vec<_>>()
        }
    }).collect::<Vec<_>>();

    Ok(scenes)
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