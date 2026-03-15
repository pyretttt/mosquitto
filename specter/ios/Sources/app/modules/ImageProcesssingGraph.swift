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
                                let option = tool.options.first
                                
                                let output = tool.process.callAsFunction(cv.SingleFrameInput(buffer))
                                await camera.inputActions
                                    .setBufferContent(output.imageBuffer.retain().takeRetainedValue())
                            }
                        })
                )
                let moduleVC = modify(CommonModuleViewController(cameraModule: cameraVC)) {
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
    cv.opencv.makeBinarizationTool(127),
]
