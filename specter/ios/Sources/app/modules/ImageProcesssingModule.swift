//
//  Playground.swift
//
//  Created by Бакулин Семен Александрович on 04.03.2026.
//

import Foundation
import core_cpp
import ios_Base

final class ImageProcessingModuleGraph {
//    var options: cv.IpToolDescription = cv.IpToolDescription(
//        name: "123",
//        options: [
//            
//        ]
//    )
    var floatOption = cv.makeIntPtr(cModify(cv.IntOption()) {
        $0.value = 32
        return $0
    })
}

extension ImageProcessingModuleGraph {
    var moduleDescription: ModuleDescription {
        ModuleDescription(
            name: "Camera playground",
            icon: "",
            searchTags: ["camera"]
        )
    }
}

extension ImageProcessingModuleGraph {
}
