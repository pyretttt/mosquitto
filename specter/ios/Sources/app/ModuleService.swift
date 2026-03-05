//
//  ModuleService.swift
//  app
//
//  Created by Бакулин Семен Александрович on 22.02.2026.
//

import Foundation

final class ModuleService: ObservableObject {
    @Published
    var modules: [ModuleDescription] = allModules
    
    var didTapModule: (ModuleDescription) -> Void
    
    init(
        modules: [ModuleDescription],
        didTapModule: @escaping (ModuleDescription) -> Void
    ) {
        self.modules = modules
        self.didTapModule = didTapModule
    }
}

let imageProcessingModule = ImageProcessingModuleGraph()

let allModules = [
    imageProcessingModule.moduleDescription,
    ModuleDescription(
        name: "SFM",
        icon: "",
        searchTags: ["sfm"]
    )

]
