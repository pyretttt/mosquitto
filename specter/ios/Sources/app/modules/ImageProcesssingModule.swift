//
//  Playground.swift
//
//  Created by Бакулин Семен Александрович on 04.03.2026.
//

import Foundation
import core_cpp
import ios_Base
import core_cpp

final class ImageProcessingModuleGraph {
    
    @MainActor weak var camera: CameraStreamViewController?
    @MainActor weak var moduleVC: CommonModuleViewController?
    
    private let binarizationTool = cv.opencv.makeBinarizationTool(127)
    
    init() {
    }
}

extension ImageProcessingModuleGraph {
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
                        didTakeAShot: { [weak self] buffer in
                            guard let self, let camera = self.camera else { assertionFailure(); return }
                            let output = binarizationTool.process.callAsFunction(cv.SingleFrameInput(buffer))
                            Task { @MainActor in
                                camera.inputActions
                                    .replaceContent(output.imageBuffer.retain().takeRetainedValue())
                            }
                        })
                )
                let moduleVC = modify(CommonModuleViewController(cameraModule: cameraVC)) {
                    $0.modalPresentationStyle = .overFullScreen
                }
                self?.camera = cameraVC
                self?.moduleVC = moduleVC
                
                return moduleVC
            }
        )
    }

}

extension ImageProcessingModuleGraph {
}

