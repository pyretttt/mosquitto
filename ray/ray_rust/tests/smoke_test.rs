#[test]
fn project_name_is_stable() {
    assert_eq!(ray_rust::project_name(), "ray_rust");
}

#[test]
fn gltf_summary_parses_minimal_document() {
    let gltf = br#"{
        "asset": {"version": "2.0"},
        "scenes": [{"nodes": [0]}],
        "nodes": [{}],
        "scene": 0
    }"#;

    let summary = ray_rust::parse_gltf_summary(gltf).expect("minimal gltf should parse");
    assert_eq!(summary.scenes, 1);
    assert_eq!(summary.nodes, 1);
    assert_eq!(summary.meshes, 0);
    assert_eq!(summary.primitives, 0);
}

#[test]
fn gltf_summary_rejects_invalid_input() {
    let invalid = br#"{"asset": {}}"#;
    assert!(ray_rust::parse_gltf_summary(invalid).is_err());
}
