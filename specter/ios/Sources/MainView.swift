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
            Tab("Modules", systemImage: "airplane.departure") {
                ModulesView()
            }
                    
            Tab("Settings", systemImage: "suitcase") {
                EmptyView()
            }
        }
        .ignoresSafeArea(.all, edges: .all)
        .environmentObject(moduleService)
    }
}
