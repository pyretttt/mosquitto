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
    
    @MainActor weak var cameraVC: CameraStreamViewController?
    @MainActor weak var moduleVC: CommonModuleViewController?
    
    private let binarizationTool = cv.opencv.makeBinarizationTool(127)
    
    init() {
    }
}

extension ImageProcessingModuleGraph {
    var moduleAPI: ModuleAPI {
        ModuleAPI(
            description: {
                ModuleDescription(
                    name: "Camera playground",
                    icon: "",
                    searchTags: ["camera"]
                )
            },
            makeScreen: { [weak self] in
                let cameraVC = CameraStreamViewController(
                    outputActions: CameraStreamViewController.OutputActions(
                        didReceiveNewBuffer: { _ in },
                        didTakeAShot: { buffer in
                            let output = binarizationTool.process.callAsFunction(cv.SingleFrameInput(buffer))
                            Task { @MainActor [weak self] in
                                self?.cameraVC?.inputActions.replaceContent(output.imageBuffer)
                            }
                        })
                )
                let moduleVC = modify(CommonModuleViewController(cameraModule: cameraVC)) {
                    $0.modalPresentationStyle = .overFullScreen
                }
                self?.cameraVC = cameraVC
                self?.moduleVC = moduleVC
                
                return moduleVC
            }
        )
    }

}

extension ImageProcessingModuleGraph {
}
