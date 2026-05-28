use crate::glib::model as glib_models;
use crate::gltf_models;
use std::rc::Rc;
use std::cell::RefCell;
use std::ffi::c_void;
use std::collections::HashMap;

pub fn map_model(gltf_model: gltf_models::Model) -> glib_models::Model {
    glib_models::Model {
        gtlf_model: gltf_model.clone(),
        scenes: gltf_model.scenes
            .iter()
            .map(map_scene)
            .collect::<Vec<_>>(),
    }
}

fn map_scene(gltf_scene: &gltf_models::Scene) -> glib_models::Scene {
    let scene = glib_models::Scene {
        gtlf_scene: gltf_scene.clone(),
        nodes: gltf_scene.nodes.iter().map(map_node).collect::<Vec<_>>(),
    };

    scene.nodes.iter().for_each(|node|{
        node.borrow().children.iter().for_each(|child| {
            child.borrow_mut().parent = Some(Rc::downgrade(node));
        });
    });

    scene
}

fn map_node(gltf_node: &Rc<RefCell<gltf_models::Node>>) -> Rc<RefCell<glib_models::Node>> {
    Rc::new(RefCell::new(glib_models::Node {
        gtlf_node: gltf_node.clone(),
        children: gltf_node.borrow().children.iter().map(map_node).collect(),
        parent: None,
        mesh: gltf_node.borrow().mesh.as_ref().map(map_mesh)
    }))
}

fn map_mesh(gltf_mesh: &Rc<gltf_models::Mesh>) -> Rc<glib_models::Mesh> {
    Rc::new(glib_models::Mesh {
        gtlf_mesh: gltf_mesh.as_ref().clone(),
        primitives: gltf_mesh.as_ref().primitives.iter().map(map_primitive).collect(),
    })
}

fn map_primitive(gltf_primitive: &gltf_models::MeshPrimitive) -> glib_models::MeshPrimitive {
    let attributes: HashMap<gltf::Semantic, glib_models::GpuAccessor> = gltf_primitive.attributes.iter()
        .map(|(semantic, accessor)| {
            (semantic.clone(), map_accessor(&accessor))
        }).collect();

    let num_positions = attributes.get(&gltf::Semantic::Positions)
        .expect("Positions attribute not found")
        .gtlf_accessor.count;

    // 0: Positions, 1: Normal, 2: Tangent, 3: TexCoord, 4: Color, 5: Joints, 6: Weights
    let mut vbos: Vec<u32> = (0..8).collect();
    let mut ebo = 0;
    let mut vao = 0;
    let has_ebo = gltf_primitive.indices.is_some();

    unsafe {
        gl::GenVertexArrays(1, &raw mut vao);
        gl::BindVertexArray(vao);

        if has_ebo {
            gl::GenBuffers(1, &raw mut ebo);
            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo);
        }

        (0..8).for_each(|idx| {
            let semantic = index_to_semantic(idx);
            let accessor = attributes.get(&semantic);
            let element_size = accessor
                .map(|acc| component_type_to_size(&acc.gtlf_accessor))
                .unwrap_or(1);
            let num = accessor.map(|acc| acc.gtlf_accessor.count).unwrap_or(num_positions);

            gl::GenBuffers(1, &raw mut vbos[idx]);
            gl::BindBuffer(gl::ARRAY_BUFFER, vbos[idx]);

            let data = accessor.map(|acc| {
                acc.gtlf_accessor.view.gltf_data.0.as_ptr().add(acc.gtlf_accessor.view.offset + acc.gtlf_accessor.offset)
            }).unwrap_or(std::ptr::null()) as *const c_void;
            gl::BufferData(
                gl::ARRAY_BUFFER,
                (element_size * num) as isize,
                data,
                gl::STATIC_DRAW
            );

            gl::VertexAttribPointer(
                idx as u32,
                accessor.map(|acc| acc.gtlf_accessor.dimensions.multiplicity()).unwrap_or(1) as i32,
                accessor.map(|acc| acc.gtlf_accessor.component_type.as_gl_enum()).unwrap_or(gl::FLOAT),
                gl::FALSE,
                element_size as i32,
                std::ptr::null() as *const c_void,
            );
            gl::EnableVertexAttribArray(idx as u32);
        });

        gl::BindVertexArray(0);
        if has_ebo {
            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, 0);
        }
        gl::BindBuffer(gl::ARRAY_BUFFER, vbos[0]);
    }

    glib_models::MeshPrimitive {
        gtlf_primitive: gltf_primitive.clone(),
        attributes: attributes,
        indices: gltf_primitive.indices.as_ref().map(map_accessor),
        material: gltf_primitive.material.as_ref().map(map_material),
    }
}

fn map_accessor(gltf_accessor: &gltf_models::GpuAccessor) -> glib_models::GpuAccessor {
    glib_models::GpuAccessor {
        gtlf_accessor: gltf_accessor.clone(),
        buffer_view: map_buffer_view(&gltf_accessor.view),
    }
}

fn map_buffer_view(gltf_buffer_view: &gltf_models::GpuBufferView) -> glib_models::BufferView {
    glib_models::BufferView {
        gtlf_buffer_view: gltf_buffer_view.clone(),
        buffer_id: 0,
    }
}

fn map_material(gltf_material: &gltf_models::Material) -> glib_models::Material {
    glib_models::Material {
        gltf_material: gltf_material.clone(),
        pbr_metallic_roughness: map_pbr_metallic_roughness(&gltf_material.pbr_metallic_roughness),
        normal_texture: gltf_material.normal_texture.as_ref().map(map_normal_texture),
        occlusion_texture: gltf_material.occlusion_texture.as_ref().map(map_occlusion_texture),
        emissive_texture: gltf_material.emissive_texture.as_ref().map(map_emissive_texture),
    }
}

fn map_pbr_metallic_roughness(gltf_pbr_metallic_roughness: &gltf_models::PbrMetallicRoughness) -> glib_models::PbrMetallicRoughness {
    glib_models::PbrMetallicRoughness {
        gltf_pbr_metallic_roughness: gltf_pbr_metallic_roughness.clone(),
        base_color_texture: gltf_pbr_metallic_roughness.base_color_texture.as_ref().map(map_texture_info),
        metallic_roughness_texture: gltf_pbr_metallic_roughness.metallic_roughness_texture.as_ref().map(map_texture_info),
    }
}

fn map_normal_texture(gltf_normal_texture: &gltf_models::NormalTexture) -> glib_models::NormalTexture {
    glib_models::NormalTexture {
        gltf_normal_texture: gltf_normal_texture.clone(),
        normal_texture: map_texture_info(&gltf_normal_texture.normal_texture),
    }
}

fn map_occlusion_texture(gltf_occlusion_texture: &gltf_models::OcclusionTexture) -> glib_models::OcclusionTexture {
    glib_models::OcclusionTexture {
        gltf_occlusion_texture: gltf_occlusion_texture.clone(),
        occlusion_texture: map_texture_info(&gltf_occlusion_texture.occlusion_texture),
    }
}

fn map_emissive_texture(gltf_emissive_texture: &gltf_models::EmissiveTexture) -> glib_models::EmissiveTexture {
    glib_models::EmissiveTexture {
        gltf_emissive_texture: gltf_emissive_texture.clone(),
        emissive_texture: map_texture_info(&gltf_emissive_texture.emissive_texture),
    }
}

fn map_texture_info(gltf_texture: &gltf_models::TextureInfo) -> glib_models::TextureInfo {
    let modes = map_texture_modes(&gltf_texture);
    let mut texture = 0;
    unsafe {
        gl::GenTextures(1, &raw mut texture);
        gl::BindTexture(gl::TEXTURE_2D, texture);

        gl::TexImage2D(
            gl::TEXTURE_2D,
            0,
            gl::RGBA as i32,
            gltf_texture.gltf_image.width as i32,
            gltf_texture.gltf_image.height as i32,
            0,
            modes.bit_format,
            gl::UNSIGNED_BYTE,
            gltf_texture.gltf_image.pixels.as_ptr() as *const c_void
        );
        if modes.mipmaps {
            gl::GenerateMipmap(gl::TEXTURE_2D);
        }

        gl::TexParameterfv(gl::TEXTURE_2D, gl::TEXTURE_BORDER_COLOR, modes.border.as_ptr() as *const f32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, modes.wrap_mode_s as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, modes.wrap_mode_t as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, modes.minifying_filter as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, modes.magnifying_filter as i32);
    }

    let texture = glib_models::TextureInfo {
        gltf_texture_info: gltf_texture.clone(),
        id: texture,
        texture_modes: modes,
    };

    unsafe { gl::BindTexture(gl::TEXTURE_2D, 0) };
    texture
}

fn map_texture_modes(_gltf_texture_info: &gltf_models::TextureInfo) -> glib_models::TextureModes {
    glib_models::TextureModes {
        wrap_mode_s: gl::REPEAT,
        wrap_mode_t: gl::REPEAT,
        magnifying_filter: gl::LINEAR,
        minifying_filter: gl::LINEAR,
        border: [1.0, 1.0, 1.0, 1.0],
        mipmaps: true,
        bit_format: gl::RGBA,
    }
}

fn index_to_semantic(index: usize) -> gltf::Semantic {
    match index {
        0 => gltf::Semantic::Positions,
        1 => gltf::Semantic::Normals,
        2 => gltf::Semantic::Tangents,
        3 => gltf::Semantic::TexCoords(0),
        4 => gltf::Semantic::TexCoords(1),
        5 => gltf::Semantic::Colors(0),
        6 => gltf::Semantic::Joints(0),
        7 => gltf::Semantic::Weights(0),
        _ => panic!("Invalid semantic index: {}", index),
    }
}

fn semantic_to_size(semantic: gltf::Semantic) -> usize {
    match semantic {
        gltf::Semantic::Positions => 3,
        gltf::Semantic::Normals => 3,
        gltf::Semantic::Tangents => 4,
        gltf::Semantic::TexCoords(0) => 2,
        gltf::Semantic::TexCoords(1) => 2,
        gltf::Semantic::Colors(0) => 4,
        gltf::Semantic::Joints(0) => 4,
        gltf::Semantic::Weights(0) => 4,
        _ => panic!("Invalid semantic: {:?}", semantic),
    }
}

fn component_type_to_size(accessor: &gltf_models::GpuAccessor) -> usize {
    accessor.dimensions.multiplicity() * accessor.component_type.size()
}