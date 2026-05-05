use std::path::Path;
use std::rc::Rc;
use std::rc::Weak;
use std::cell::RefCell;
use gltf::accessor::{DataType, Dimensions};
use std::collections::HashMap;
use wgpu::util::DeviceExt;


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

#[derive(Clone)]
pub struct Material {
    pub name: String,
    pub pbr_metallic_roughness: Option<PbrMetallicRoughness>,
    pub normal_texture: Option<NormalTexture>,
    pub occlusion_texture: Option<OcclusionTexture>,
    pub emissive_texture: Option<EmissiveTexture>,
    pub emissive_factor: [f32; 3],
}

#[derive(Clone)]
pub struct PbrMetallicRoughness {
    pub base_color_texture: Option<Rc<TextureInfo>>,
    pub base_color_factor: [f32; 4],
    pub base_color_tex_coord: usize,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub metallic_roughness_texture: Option<Rc<TextureInfo>>,
}

#[derive(Clone)]
pub struct NormalTexture {
    pub scale: f32,
    pub normal_texture: Option<Rc<TextureInfo>>,
    pub tex_coord: usize,
}

#[derive(Clone)]
pub struct OcclusionTexture {
    pub strength: f32,
    pub occlusion_texture: Option<Rc<TextureInfo>>,
    pub tex_coord: usize,
}

#[derive(Clone)]
pub struct EmissiveTexture {
    pub emissive_texture: Option<Rc<TextureInfo>>,
    pub tex_coord: usize,
}

pub fn load_gltf(path: &Path) -> Result<GLTF, gltf::Error> {
    let (document, buffers, textures) = gltf::import(path)?;
    Ok(GLTF { document, buffers, textures })
}

pub fn make_wgpu_scenes(gltf: &GLTF, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Vec<Scene>, wgpu::Error> {
    let textures = private::load_textures(gltf, device, queue);

    let mut buffers_usages: HashMap<usize, wgpu::BufferUsages> = HashMap::new();
    gltf.document.meshes().for_each(|mesh: gltf::Mesh<'_>| {
        mesh.primitives().for_each(|primitive| {
            primitive.attributes().for_each(|(_, attribute_accessor)| {
                let Some(buffer_view) = attribute_accessor.view() else {
                    return;
                };
                *buffers_usages.entry(buffer_view.buffer().index())
                    .or_insert(wgpu::BufferUsages::empty()) |= wgpu::BufferUsages::VERTEX;
            });

            let Some(indices_accessor) = primitive.indices() else {
                return;
            };
            let Some(index_buffer_view) = indices_accessor.view() else {
                return;
            };
            *buffers_usages.entry(index_buffer_view.buffer().index())
                .or_insert(wgpu::BufferUsages::empty()) |= wgpu::BufferUsages::INDEX;
        });
    });

    let mut buffers: HashMap<usize, wgpu::Buffer> = HashMap::new();
    gltf.document.buffers().for_each(|buffer| {
        let buffer_index = buffer.index();
        if buffers.get(&buffer_index).is_some() { return; };

        if let Some(buffer_usages) = buffers_usages.get(&buffer_index) {
            let buffer_data = &gltf.buffers[buffer_index];
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Buffer({})", buffer_index)),
                contents: buffer_data.0.as_slice(),
                usage: *buffer_usages,
            });

            buffers.insert(buffer_index, buffer);
        } else {
            println!("Buffer({}) not used", buffer_index);
        }
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

    let meshes: Vec<Rc<Mesh>> = gltf.document.meshes().map(|mesh| {
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
                ),
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

mod private {
    use super::HashMap;
    use super::DeviceExt;

    const PBR_BASE_COLOR_TEXTURE_KEY: &str = "pbr_base_color";
    const NORMAL_TEXTURE_KEY: &str = "normal";
    const OCCLUSION_TEXTURE_KEY: &str = "occlusion";
    const EMISSIVE_TEXTURE_KEY: &str = "emissive";
    const METALLIC_ROUGHNESS_TEXTURE_KEY: &str = "metallic_roughness";

    #[derive(Clone, Debug, Hash, PartialEq, Eq)]
    struct TextureKey {
        index: usize,
        tex_type: String
    }

    struct SamplerWrapper<'a>(gltf::texture::Sampler<'a>);

    impl<'a> SamplerWrapper<'a> {
        fn make_desriptor(&self, label: &'a str) -> wgpu::SamplerDescriptor<'a> {
            wgpu::SamplerDescriptor {
                label: Some(label),
                address_mode_u: map_address_mode(&self.0.wrap_s()),
                address_mode_v: map_address_mode(&self.0.wrap_t()),
                mag_filter: self.0.mag_filter().as_ref().map(map_mag_filter).unwrap_or(wgpu::SamplerDescriptor::default().mag_filter),
                min_filter: self.0.min_filter().as_ref().map(map_min_filter).unwrap_or(wgpu::SamplerDescriptor::default().min_filter),
                ..Default::default()
            }
        }
    }

    pub fn load_textures(gltf: &super::GLTF, device: &wgpu::Device, queue: &wgpu::Queue) -> HashMap<TextureKey, super::TextureInfo> {
        let mut textures: HashMap<TextureKey, super::TextureInfo> = HashMap::new();
        gltf.document.materials().for_each(|material| {
            if let Some(base_color_texture) = material.pbr_metallic_roughness().base_color_texture() {
                let key = TextureKey {
                    index: base_color_texture.texture().index(),
                    tex_type: PBR_BASE_COLOR_TEXTURE_KEY.to_owned(),
                };
                if textures.get(&key).is_some() { return; }

                let texture_info = load_texture_info(gltf, device, queue, &base_color_texture.texture(), wgpu::TextureFormat::Rgba8UnormSrgb);
                textures.insert(key, texture_info);
            }

            if let Some(normal_texture) = material.normal_texture() {
                let key = TextureKey {
                    index: normal_texture.texture().index(),
                    tex_type: NORMAL_TEXTURE_KEY.to_owned(),
                };
                if textures.get(&key).is_some() { return; }

                let texture_info = load_texture_info(gltf, device, queue, &normal_texture.texture(), wgpu::TextureFormat::Rgba8Unorm);
                textures.insert(key, texture_info);
            }

            if let Some(occlusion_texture) = material.occlusion_texture() {
                let key = TextureKey {
                    index: occlusion_texture.texture().index(),
                    tex_type: OCCLUSION_TEXTURE_KEY.to_owned(),
                };
                if textures.get(&key).is_some() { return; }

                let texture_info = load_texture_info(gltf, device, queue, &occlusion_texture.texture(), wgpu::TextureFormat::Rgba8Unorm);
                textures.insert(key, texture_info);
            }

            if let Some(emissive_texture) = material.emissive_texture() {
                let key = TextureKey {
                    index: emissive_texture.texture().index(),
                    tex_type: EMISSIVE_TEXTURE_KEY.to_owned(),
                };
                if textures.get(&key).is_some() { return; }

                let texture_info = load_texture_info(gltf, device, queue, &emissive_texture.texture(), wgpu::TextureFormat::Rgba8UnormSrgb);
                textures.insert(key, texture_info);
            }

            if let Some(metallic_roughness_texture) = material.pbr_metallic_roughness().metallic_roughness_texture() {
                let key = TextureKey {
                    index: metallic_roughness_texture.texture().index(),
                    tex_type: METALLIC_ROUGHNESS_TEXTURE_KEY.to_owned(),
                };
                if textures.get(&key).is_some() { return; }

                let texture_info = load_texture_info(gltf, device, queue, &metallic_roughness_texture.texture(), wgpu::TextureFormat::Rgba8Unorm);
                textures.insert(key, texture_info);
            }
        });

        textures
    }

    fn load_texture_info(
        gltf: &super::GLTF,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &gltf::texture::Texture<'_>,
        format: wgpu::TextureFormat
    ) -> super::TextureInfo {
        let texture_data = gltf.textures.get(texture.index()).expect("Failed to obtain texture");
        let size = wgpu::Extent3d {
            width: texture_data.width,
            height: texture_data.height,
            depth_or_array_layers: 1,
        };
        let wgpu_texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some(&format!("Texture({})", texture.index())),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            Default::default(),
            texture_data.pixels.as_slice(),
        );
        let texture_view = wgpu_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler_wrapper = SamplerWrapper(texture.sampler());
        let texture_sampler = device.create_sampler(&sampler_wrapper.make_desriptor(
            sampler_wrapper.0.name().unwrap_or("texture_sampler")
        ));

        super::TextureInfo {
            texture: wgpu_texture,
            view: texture_view,
            sampler: texture_sampler,
        }
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

    fn map_address_mode(mode: &gltf::texture::WrappingMode) -> wgpu::AddressMode {
        match mode {
            gltf::texture::WrappingMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
            gltf::texture::WrappingMode::Repeat => wgpu::AddressMode::Repeat,
            gltf::texture::WrappingMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
        }
    }

    fn map_mag_filter(mode: &gltf::texture::MagFilter) -> wgpu::FilterMode {
        match mode {
            gltf::texture::MagFilter::Nearest => wgpu::FilterMode::Nearest,
            gltf::texture::MagFilter::Linear => wgpu::FilterMode::Linear,
        }
    }

    fn map_min_filter(mode: &gltf::texture::MinFilter) -> wgpu::FilterMode {
        match mode {
            gltf::texture::MinFilter::Nearest => wgpu::FilterMode::Nearest,
            gltf::texture::MinFilter::Linear => wgpu::FilterMode::Linear,
            gltf::texture::MinFilter::NearestMipmapNearest => wgpu::FilterMode::Nearest,
            gltf::texture::MinFilter::LinearMipmapNearest => wgpu::FilterMode::Linear,
            gltf::texture::MinFilter::NearestMipmapLinear => wgpu::FilterMode::Nearest,
            gltf::texture::MinFilter::LinearMipmapLinear => wgpu::FilterMode::Linear,
        }
    }
}
