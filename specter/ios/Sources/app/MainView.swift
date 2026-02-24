//
//  MainView.swift
//  app
//
//  Created by Бакулин Семен Александрович on 21.02.2026.
//

import Foundation
import SwiftUI

struct MainView: View {
    
    let moduleService: ModuleService
    
    var body: some View {
        TabView {
            Tab("Modules", systemImage: "camera") {
                ModulesView()
            }
                    
            Tab("Settings", systemImage: "settings") {
                EmptyView()
            }
        }
        .ignoresSafeArea(.all, edges: .all)
        .environmentObject(moduleService)
    }
}
