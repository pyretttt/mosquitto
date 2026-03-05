//
//  Playground.swift
//
//  Created by Бакулин Семен Александрович on 04.03.2026.
//

import Foundation
import core_cpp

final class ImageProcessingModuleGraph {
    var options: ip_tool.IpToolDescription = ip_tool.IpToolDescription.init(
        name: "123",
        options: [],
        selectors: []
    )
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
