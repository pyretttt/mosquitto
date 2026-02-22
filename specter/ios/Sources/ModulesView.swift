//
//  ModulesView.swift
//  app
//
//  Created by Бакулин Семен Александрович on 22.02.2026.
//

import Foundation
import SwiftUI

struct ModulesView: View {
    
    @EnvironmentObject var moduleService: ModuleService

    @State private var searchText: String = ""

    private var filteredItems: [Module] {
        let search = searchText.trimmingCharacters(in: .whitespaces).lowercased()
        if search.isEmpty {
            return moduleService.modules
        }
        return moduleService.modules
            .filter {
                $0.name.localizedCaseInsensitiveContains(search)
                || $0.searchTags.contains(search)
            }
    }
    
    var body: some View {
        NavigationStack {
            List(filteredItems, id: \.self) { item in
                Text(item.name)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .listRowSeparator(.hidden)
            }
            .listStyle(.plain)
            .navigationBarTitleDisplayMode(.inline)
            .navigationTitle("Modules")
        }
        .searchable(text: $searchText, prompt: "Pick module")
        .toolbarBackground(.visible, for: .navigationBar)
    }
}
