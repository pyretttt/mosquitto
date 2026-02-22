//
//  MainView.swift
//  app
//
//  Created by Бакулин Семен Александрович on 21.02.2026.
//

import Foundation
import SwiftUI

struct MainView: View {
    // Sample data source; replace with real items as needed.
    private let items: [String] = [
        "Apples",
        "Bananas",
        "Cherries",
        "Dates",
        "Elderberries",
        "Figs",
        "Grapes"
    ]

    @State private var searchText: String = ""

    private var filteredItems: [String] {
        guard !searchText.trimmingCharacters(in: .whitespaces).isEmpty else { return items }
        return items.filter { $0.localizedCaseInsensitiveContains(searchText) }
    }

    var body: some View {
        TabView {
            Tab("Modules", systemImage: "airplane.departure") {
                NavigationStack {
                    VStack(spacing: 12) {
                        // List centered horizontally; it naturally fills vertical space below the top bar.
                        List(filteredItems, id: \.self) { item in
                            Text(item)
                                .frame(maxWidth: .infinity, alignment: .center)
                        }
                        .listStyle(.insetGrouped)
                        
                        
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                    .background(Color(.systemBackground))
                }
                .searchable(
                    text: $searchText,
                    prompt: "Look for something"
                )
            }
                    
            Tab("Settings", systemImage: "suitcase") {
                EmptyView()
            }
        }
        
    }
}
