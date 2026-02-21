// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "SpecterSPMDeps",
    platforms: [
        .iOS(.v15)
    ],
    products: [],
    dependencies: [
        .package(url: "https://github.com/SnapKit/SnapKit.git", from: "5.7.1")
    ],
    targets: []
)
