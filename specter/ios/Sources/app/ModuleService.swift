//
//  ModuleService.swift
//  app
//
//  Created by Бакулин Семен Александрович on 22.02.2026.
//

import UIKit

final class ModuleService: ObservableObject {
    @Published
    var modules: [ModuleDescription] = allModules.map(\.description)
    
    var didTapModule: (ModuleDescription) -> Void
    
    init(
        didTapModule: @escaping (ModuleDescription) -> Void
    ) {
        self.didTapModule = didTapModule
    }
}

struct ModuleHolder {
    var description: () -> ModuleDescription
    var makeScreen: () -> UIViewController
}

let imageProcessingModule = ImageProcessingGraph()

let allModules = [
    imageProcessingModule.moduleAPI
]
