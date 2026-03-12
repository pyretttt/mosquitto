# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Specter is an iOS camera app with modular image processing, built using Swift, C++, and Rust with Bazel as the build system.

## Build Commands

```bash
# Setup (install Bazelisk and Rust via mise)
mise install

# Generate Xcode project and open it
bazelisk run //:xcodeproj && xed specter.xcodeproj

# Build targets
bazelisk build //ios:app              # iOS app
bazelisk build //core/cpp:core_cpp    # C++ library
bazelisk build //core/rust:core_rust  # Rust library

# Run C++ tests
bazel test //core/cpp:core_cpp_test

# Update Swift package dependencies
bazel mod tidy

# Clean build
bazel clean && rm -rf specter.xcodeproj
```

## Architecture

```
Specter
├── ios/                    # SwiftUI + UIKit iOS app
│   └── Sources/
│       ├── app/            # Main app, views, camera pipeline, modules
│       ├── base/           # Utilities (modify(), base classes)
│       └── shaders/        # Metal GPU shaders
│
└── core/                   # Shared native libraries
    ├── cpp/                # OpenCV image processing (C++23)
    ├── rust/               # Rust FFI exports
    └── bridge/             # C headers for Rust FFI
```

### Camera Pipeline Flow
1. `CameraStreamViewController` captures frames via AVCaptureSession
2. `CVImageBuffer` converted to OpenCV `Mat` in C++ layer (`core/cpp/src/ios/cv.cpp`)
3. Processing modules apply transformations
4. Results displayed back in Swift UI

### Module System
- `ModuleDescription` defines module metadata
- `ModuleService` manages available modules
- Modules implement `ImageProcessingModuleGraph` protocol
- Add new modules in `ios/Sources/app/modules/`

### Cross-Language Interop
- **Swift ↔ C++**: Direct interop via `swift_interop_hint` in BUILD.bazel. Types defined in `core/cpp/include/ios/cv.hpp`
- **Swift ↔ Rust**: C FFI via `core/bridge/bridge.h`, Rust exports in `core/rust/src/lib.rs`

## Code Style (from ios/AGENTS.md)

- Use `modify()` for fluent object configuration:
  ```swift
  let view = modify(UIView()) {
      $0.backgroundColor = .red
  }
  ```
- Prefer computed properties over stored properties
- Use structs with closures over protocols
- Function names should describe actions, not implementation

## Key Dependencies

- **Bazel 7.4.1** (via Bazelisk)
- **OpenCV 4.13.0** - Computer vision
- **Ceres Solver 2.2.0** - Optimization
- **Eigen 3.4.0** - Linear algebra
- **SnapKit 5.7.1** - Auto-layout DSL
