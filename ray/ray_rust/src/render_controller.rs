use crate::gltf;
use std::collections::HashMap;

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

impl Vertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 9]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 11]>() as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 13]>() as wgpu::BufferAddress,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 19]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

// pub fn map_semantic(semantic: &gltf::Semantic) -> u32 {
//     match semantic {
//         gltf::Semantic::Positions => 0,
//         gltf::Semantic::Normals => 1,
//         gltf::Semantic::Tangents => 2,
//         gltf::Semantic::TexCoords(x) if *x < 2 => 3 + x,
//         gltf::Semantic::Colors(x) if *x == 0 => 5,
//         gltf::Semantic::Joints(x) if *x == 0 => 6,
//         gltf::Semantic::Weights(x) if *x == 0 => 7,
//         _ => panic!("Unsupported semantic: {:?}", semantic),
//     }
// }

pub struct RenderController {
    pub render_pipelines: HashMap<String, wgpu::RenderPipeline>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl RenderController {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, model: &gltf::Model) -> Self {
        Self {
            render_pipelines: HashMap::from([
                ("prb".to_owned(), helpers::make_prb_render_pipeline(device, shader, vertex_layout))
            ]),
            device: device.clone(),
            queue: queue.clone(),
        }
    }
}

impl RenderController {
    pub fn render(&self, model: &gltf::Model) -> Result<(), String> {
        let scene = model.scenes
            .get(model.selected_scene.unwrap_or(0)).expect("Selected scene not found");

        self.render_pipelines.keys().for_each(|key| {
            scene.nodes.iter().for_each(|node| {
                let Some(mesh) = &node.borrow().mesh else { return; };
                mesh.primitives.iter()
                    .filter(|primitive| primitive.pipeline_key() == *key)
                    .for_each(|primitive| {});
                // mesh.primitives.iter().filter()
            });
        });
        Ok(())
    }


}

impl gltf::MeshPrimitive {
    pub fn pipeline_key(&self) -> String {
        "prb".to_owned()
    }
}

mod helpers {
    pub fn make_prb_render_pipeline(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        vertex_layout: &wgpu::VertexBufferLayout
    ) -> wgpu::RenderPipeline {
        let vertex_layout = Vertex::desc();
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[crate::model::ModelVertex::desc(), InstanceRaw::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState { // 3.
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState { // 4.
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // 1.
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, // 2.
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: crate::texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less), // 1.
                stencil: wgpu::StencilState::default(), // 2.
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        })
    }
}