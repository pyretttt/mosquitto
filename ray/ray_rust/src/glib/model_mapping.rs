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
    }))
}