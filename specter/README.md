# Specter

Bazel-based iOS-first app skeleton with a Swift app and shared C++ and Rust libraries.

## Quick Start
1. `mise install`
2. `bazel build //ios/App:App`
3. `bazel build //shared/cpp:hello_cpp`
4. `bazel build //shared/rust:hello_rust`

## Notes
- This repo uses Bzlmod (`MODULE.bazel`) instead of a `WORKSPACE`.
- iOS builds require Xcode and the command line tools installed (`xcode-select -p` should succeed).
- Update versions in `MODULE.bazel` and `.bazelversion` as needed.
