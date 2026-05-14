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

pub fn make_render_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    vertex_layout: &wgpu::VertexBufferLayout
) -> wgpu::RenderPipeline {
    let vertex_layout = Vertex::desc();
    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("GLTF Render Pipeline Layout"),
        entries: &[
            wgpu::PipelineLayoutDescriptorEntry::VertexBuffer(vertex_layout)
        ],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            entries: &[wgpu::PipelineLayoutDescriptorEntry::VertexBuffer(vertex_layout)],
        }),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: "main",
        },
    })
}