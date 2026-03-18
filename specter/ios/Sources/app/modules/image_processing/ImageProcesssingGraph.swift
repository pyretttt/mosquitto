//
//  Playground.swift
//
//  Created by Бакулин Семен Александрович on 04.03.2026.
//

import Foundation
import SwiftUI
import UIKit
import core_cpp
import ios_Base

final class ImageProcessingGraph {
    init() {}
}

extension ImageProcessingGraph {
    var moduleAPI: ModuleAPI {
        ModuleAPI(
            description: ModuleDescription(
                name: "Camera playground",
                icon: "",
                searchTags: ["camera", "image", "processing"]
            ),
            makeScreen: {
                makeToolSelectionScreen(tools: tools)
            }
        )
    }
}

@MainActor
func makeToolSelectionScreen(tools: [cv.SingleFrameIpTool]) -> UIViewController {
    weak var view: UIViewController?
    let vc = modify(
        UIHostingController(
            rootView: ImageProcessingToolSelectionView(
                tools: tools.map(\.name).map(String.init),
                onClose: {
                    view?.dismiss(animated: true)
                },
                onSelectTool: { selectedIndex in
                    view?.present(
                        makeModuleScreen(tool: tools[selectedIndex]),
                        animated: true
                    )
                }
            )
        )
    ) {
        $0.modalPresentationStyle = .fullScreen
        view = $0
    }
    return vc
}

@MainActor
func makeModuleScreen(tool: cv.SingleFrameIpTool) -> UIViewController {
    weak var camera: CameraStreamViewController?
    let cameraVC = modify(CameraStreamViewController(
        outputActions: CameraStreamViewController.OutputActions(
            didReceiveNewBuffer: { _ in },
            didTakeAShot: { buffer in
                guard let camera else { assertionFailure(); return }
                Task.detached(priority: .userInitiated) {
                    let output = tool
                        .process
                        .callAsFunction(cv.SingleFrameInput(buffer))
                    await camera.inputActions
                        .setBufferContent(output.imageBuffer.retain().takeRetainedValue())
                }
            }
        )
    )) {
        camera = $0
    }

    return modify(
        CommonModuleViewController(
            cameraModule: cameraVC,
            optionsProvider: {
                buildOptionModels(from: tool.options)
            }
        )
    ) {
        $0.modalPresentationStyle = .overFullScreen
    }
}

private let tools: [cv.SingleFrameIpTool] = [
    cv.opencv.makeBinarizationTool(),
]

func buildOptionModels(
    from options: cv.OptionList
) -> [OptionModel] {
    options.map {
        let name = String($0.pointee.name)
        return switch $0.pointee.kind {
        case cv.OptionKind.Int:
            OptionModel.int(name: name, ptr: cv.asInt($0))
        case .Float:
            OptionModel.float(name: name, ptr: cv.asFloat($0))
        case .Bool:
            OptionModel.bool(name: name, ptr: cv.asBool($0))
        case .String:
            OptionModel.string(name: name, ptr: cv.asString($0))
        case .MultiString:
            OptionModel.multiString(name: name, ptr: cv.asMultiString($0))
        case .MultiInteger:
            OptionModel.multiInt(name: name, ptr: cv.asMultiInteger($0))
        case .MultiFloat:
            OptionModel.multiFloat(name: name, ptr: cv.asMultiFloat($0))
        @unknown default:
            fatalError()
        }
    }
}
