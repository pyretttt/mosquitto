mod gltf_loader;

pub use gltf_loader::{load_gltf_summary, parse_gltf_summary, GltfSummary};

pub fn project_name() -> &'static str {
    "ray_rust"
}
