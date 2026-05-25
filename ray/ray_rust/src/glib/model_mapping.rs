use crate::glib::model as glib_models;
use crate::gltf_models;
use std::rc::Rc;
use std::cell::RefCell;

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
    glib_models::Scene {
        gtlf_scene: gltf_scene.clone(),
        nodes: gltf_scene.nodes.iter().map(map_node).collect::<Vec<_>>(),
    }
}

fn map_node(gltf_node: &Rc<RefCell<gltf_models::Node>>) -> Rc<RefCell<glib_models::Node>> {
    Rc::new(RefCell::new(glib_models::Node {
        gtlf_node: gltf_node.clone(),
        children: gltf_node.borrow().children.iter().map(map_node).collect(),
        parent: gltf_node.borrow().parent.as_ref().map(|parent| {
            match parent.upgrade() {
                Some(parent) => Some(Rc::downgrade(&map_node(&parent))),
                None => None,
            }
        }).flatten(),
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
    glib_models::MeshPrimitive {
        gtlf_primitive: gltf_primitive.clone(),
        attributes: gltf_primitive.attributes.iter().map(|(semantic, accessor)| {
            (semantic.clone(), map_accessor(&accessor))
        }).collect(),
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

fn map_texture_info(gltf_texture: &gltf_models::TextureInfo) -> glib_models::TextureInfo {
    glib_models::TextureInfo {
        gltf_texture_info: gltf_texture.clone(),
        id: 0,
        texture_modes: map_texture_modes(&gltf_texture),
    }
}

fn map_texture_modes(_gltf_texture_info: &gltf_models::TextureInfo) -> glib_models::TextureModes {
    glib_models::TextureModes {
        wrap_mode_s: gl::REPEAT,
        wrap_mode_t: gl::REPEAT,
        magnifying_filter: gl::LINEAR,
        minifying_filter: gl::LINEAR,
        border: [1.0, 1.0, 1.0, 1.0],
        mipmaps: true,
        bit_format: gl::RGB,
    }
}