
pub use gltf_loader::{load_gltf_summary, parse_gltf_summary, GltfSummary};

mod gltf_loader;
pub mod common;

pub fn project_name() -> &'static str {
    "ray_rust"
}
