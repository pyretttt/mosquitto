use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GltfSummary {
    pub scenes: usize,
    pub nodes: usize,
    pub meshes: usize,
    pub primitives: usize,
    pub materials: usize,
    pub animations: usize,
}

fn summarize(document: &gltf::Document) -> GltfSummary {
    let scenes = document.scenes().count();
    let nodes = document.nodes().count();
    let meshes = document.meshes().count();
    let primitives = document
        .meshes()
        .map(|mesh| mesh.primitives().count())
        .sum();
    let materials = document.materials().count();
    let animations = document.animations().count();

    GltfSummary {
        scenes,
        nodes,
        meshes,
        primitives,
        materials,
        animations,
    }
}

pub fn parse_gltf_summary(gltf_json: &[u8]) -> Result<GltfSummary, String> {
    let gltf = gltf::Gltf::from_slice(gltf_json).map_err(|err| err.to_string())?;
    Ok(summarize(&gltf.document))
}

pub fn load_gltf_summary(path: &Path) -> Result<GltfSummary, String> {
    let bytes = std::fs::read(path).map_err(|err| err.to_string())?;
    parse_gltf_summary(&bytes)
}
