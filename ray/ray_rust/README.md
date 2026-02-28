# ray_rust

Step 1 and 2 scaffold for Rust migration:

1. Bazel build workspace for Rust targets.
2. `mise` configuration for tool installation (`bazel`, `buildifier`, `rust`).

## Tool setup (Mise)

```bash
cd ray_rust
mise install
```

Optional task aliases:

```bash
mise run setup
mise run build
mise run test
mise run run
```

## Bazel targets

```bash
bazel build //:app
bazel test //:smoke_test
bazel run //:app
```

## GLTF Loading

The app can parse a GLTF/GLB file at startup and print a summary:

```bash
RAY_GLTF=/absolute/path/to/model.gltf bazel run //:app
```

## Files added

- `MODULE.bazel`: Bazel module + `rules_rust` toolchain registration.
- `BUILD.bazel`: `rust_library`, `rust_binary`, `rust_test` targets.
- `.mise.toml`: Tool dependencies and helper tasks.
- `src/`, `tests/`: Minimal Rust scaffold to validate wiring.
