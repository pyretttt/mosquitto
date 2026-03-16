//
//  Playground.swift
//
//  Created by Бакулин Семен Александрович on 04.03.2026.
//

import Foundation
import core_cpp
import ios_Base
import core_cpp

final class ImageProcessingGraph {
    
    @MainActor weak var camera: CameraStreamViewController?
    @MainActor weak var moduleVC: CommonModuleViewController?
    
    private let selectedTool = 0
    
    init() {
        
    }
}

extension ImageProcessingGraph {
    var moduleAPI: ModuleAPI {
        ModuleAPI(
            description: ModuleDescription(
                name: "Camera playground",
                icon: "",
                searchTags: ["camera"]
            ),
            makeScreen: { [weak self] in
                let cameraVC = CameraStreamViewController(
                    outputActions: CameraStreamViewController.OutputActions(
                        didReceiveNewBuffer: { _ in },
                        didTakeAShot: { buffer in
                            Task.detached { [weak self] in
                                guard let self, let camera = await self.camera else { assertionFailure(); return }
                                let tool = tools[self.selectedTool]
                                let output = tool.process.callAsFunction(cv.SingleFrameInput(buffer))
                                await camera.inputActions
                                    .setBufferContent(output.imageBuffer.retain().takeRetainedValue())
                            }
                        })
                )
                let moduleVC = modify(CommonModuleViewController(
                    cameraModule: cameraVC,
                    optionsProvider: {
                        guard let self else { return [] }
                        return buildOptionModels(
                            from: tools[self.selectedTool].options
                        )
                    }
                )) {
                    $0.modalPresentationStyle = .overFullScreen
                }
                modify(self) {
                    $0?.camera = cameraVC
                    $0?.moduleVC = moduleVC
                }
                
                return moduleVC
            }
        )
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
