use crate::gltf;
use std::collections::HashMap;

pub struct RenderController {
    pub render_pipelines: HashMap<String, wgpu::RenderPipeline>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::SurfaceConfiguration,
}

impl RenderController {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &gltf::Model,
        surface_config: &wgpu::SurfaceConfiguration
    ) -> Self {
        let mut this = Self {
            render_pipelines: HashMap::new(),
            device: device.clone(),
            queue: queue.clone(),
            surface: surface_config.clone(),
        };
        this.render_pipelines = HashMap::from([
            ("prb".to_owned(), helpers::make_prb_render_pipeline(&this))
        ]);
        this
    }
}

impl RenderController {
    pub fn render(&self, model: &gltf::Model, view: &wgpu::TextureView, clear_color: wgpu::Color, depth_texture: &crate::texture::Texture) -> Result<(), String> {
        let scene = model.scenes
            .get(model.selected_scene.unwrap_or(0)).expect("Selected scene not found");

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[
                // This is what @location(0) in the fragment shader targets
                Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_texture.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });

        self.render_pipelines.keys().for_each(|key| {
            scene.nodes.iter().for_each(|node| {
                let Some(mesh) = &node.borrow().mesh else { return; };
                mesh.primitives.iter()
                    .filter(|primitive| primitive.pipeline_key() == *key)
                    .for_each(|primitive| {
                        self.render_primitive(primitive, &mut render_pass);
                    });
            });
        });
        Ok(())
    }

    pub fn render_primitive(&self, primitive: &gltf::MeshPrimitive, render_pass: & mut wgpu::RenderPass<'_>) {
        let pipeline = self.render_pipelines.get(&primitive.pipeline_key()).unwrap();
        render_pass.set_pipeline(pipeline);
        primitive.attributes.iter().for_each(|(semantic, accessor)| {
            let Some(slot) = helpers::map_semantic(semantic) else { return };
            let byte_offset = accessor.view.offset + accessor.offset;
            render_pass.set_vertex_buffer(
                slot,
                accessor.view.buf.slice(byte_offset..),
            );
        });
        if let Some(indices) = &primitive.indices {
            let offset = indices.offset + indices.view.offset;
            render_pass.set_index_buffer(indices.view.buf.slice(offset..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..indices.count as u32, 0, 0..1);
        } else {
            render_pass.draw(0..primitive.attributes.len() as u32, 0..1);
        }
    }
}

impl gltf::MeshPrimitive {
    pub fn pipeline_key(&self) -> String {
        "prb".to_owned()
    }
}

mod helpers {
    use super::*;
    use ::gltf::Semantic;

    pub fn map_semantic(semantic: &Semantic) -> Option<u32> {
        match semantic {
            Semantic::Positions => Some(0),
            Semantic::Normals => Some(1),
            Semantic::Tangents => Some(2),
            Semantic::TexCoords(x) if *x < 2 => Some(3 + x),
            Semantic::Colors(x) if *x == 0 => Some(5),
            Semantic::Joints(x) if *x == 0 => Some(6),
            Semantic::Weights(x) if *x == 0 => Some(7),
            _ => None,
        }
    }

    const PBR_VERTEX_BUFFER_LAYOUTS: [wgpu::VertexAttribute; 8] = [
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 1 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 2 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 3 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 4 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 5 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 6 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 7 },
    ];

    const fn separate_vertex_layout<'a>(
        attribute: &'a [wgpu::VertexAttribute],
        array_stride: wgpu::BufferAddress,
    ) -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: attribute,
        }
    }

    pub fn make_prb_render_pipeline(
        render_controller: &RenderController
    ) -> wgpu::RenderPipeline {
        let render_pipeline_layout = render_controller.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[

            ],
            immediate_size: 0,
        });

        let shader = render_controller.device.create_shader_module(
            wgpu::include_wgsl!("shaders/pbr.wgsl")
        );

        render_controller.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex_main"),
                buffers: &[
                    separate_vertex_layout(&PBR_VERTEX_BUFFER_LAYOUTS.get(0..1).expect("wrong pbr buffers"), 12),
                    separate_vertex_layout(&PBR_VERTEX_BUFFER_LAYOUTS.get(1..2).expect("wrong pbr buffers"), 12),
                    separate_vertex_layout(&PBR_VERTEX_BUFFER_LAYOUTS.get(2..3).expect("wrong pbr buffers"), 12),
                    separate_vertex_layout(&PBR_VERTEX_BUFFER_LAYOUTS.get(3..4).expect("wrong pbr buffers"), 8),
                    separate_vertex_layout(&PBR_VERTEX_BUFFER_LAYOUTS.get(4..5).expect("wrong pbr buffers"), 8),
                    separate_vertex_layout(&PBR_VERTEX_BUFFER_LAYOUTS.get(5..6).expect("wrong pbr buffers"), 12),
                    separate_vertex_layout(&PBR_VERTEX_BUFFER_LAYOUTS.get(6..7).expect("wrong pbr buffers"), 12),
                    separate_vertex_layout(&PBR_VERTEX_BUFFER_LAYOUTS.get(7..8).expect("wrong pbr buffers"), 12),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState { // 3.
                module: &shader,
                entry_point: Some("fragment_main"),
                targets: &[Some(wgpu::ColorTargetState { // 4.
                    format: render_controller.surface.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // 1.
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
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