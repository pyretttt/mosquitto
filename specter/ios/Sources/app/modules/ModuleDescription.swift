//
//  ModuleDescription.swift
//
//  Created by Бакулин Семен Александрович on 04.03.2026.
//

import UIKit

struct ModuleDescription: Hashable {
    var name: String
    var icon: String
    var searchTags: [String]
}

struct ModuleAPI {
    var description: ModuleDescription
    var makeScreen: @Sendable @MainActor () -> UIViewController
}
