//
//  ModuleService.swift
//  app
//
//  Created by Бакулин Семен Александрович on 22.02.2026.
//

import Foundation

struct Module: Hashable {
    var name: String
    var icon: String
    var searchTags: [String]
}

final class ModuleService: ObservableObject {
    @Published
    var modules: [Module] = allModules
    
    var didTapModule: (Module) -> Void
    
    init(
        modules: [Module],
        didTapModule: @escaping (Module) -> Void
    ) {
        self.modules = modules
        self.didTapModule = didTapModule
    }
}

let allModules = [
    Module(
        name: "Camera playground",
        icon: "",
        searchTags: ["camera"]
    ),
    Module(
        name: "SFM",
        icon: "",
        searchTags: ["sfm"]
    )

]
