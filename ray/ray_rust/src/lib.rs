
pub use gltf_loader::{load_gltf_summary, parse_gltf_summary, GltfSummary};

mod gltf_loader;
pub mod common;
pub mod vertex;
pub mod texture;
pub mod camera_controller;

pub fn project_name() -> &'static str {
    "ray_rust"
}
