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

    private var filteredItems: [ModuleDescription] {
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
        // Frezes view start, and there're some hangs, so don't use it for now
            NavigationStack {
                List(filteredItems, id: \.self) { item in
                    Text(item.name)
                        .font(.system(size: 18, weight: .regular, design: .default))
                        .foregroundStyle(textColor)
                        .frame(maxWidth: .infinity, alignment: .center)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 6)
                        .listRowSeparator(.hidden)
                        .onTapGesture {
                            moduleService.didTapModule(item)
                        }
                        .listRowBackground(
                            tableBackgroundColor.opacity(0.95)
                        )
                        .hoverEffect(.highlight)
                }
                .listStyle(.insetGrouped)
                .navigationBarTitleDisplayMode(.inline)
                .scrollContentBackground(.hidden)
                .navigationTitle("Modules")
                .containerBackground(for: .navigation) {
//                    TimelineView(.periodic(from: .now, by: 1.0 / 30.0)) { context in
//                        let time = Float(refDate.timeIntervalSinceNow)
//                        Rectangle()
//                            .visualEffect { view, proxy in
//                                view.colorEffect(
//                                    ShaderLibrary.identity(
//                                        proxy.frame(in: .global).shaderArg,
//                                        .float(time)
//                                    )
//                                )
//                            }
//                    }
                    Color.white
                    .ignoresSafeArea()
                }
            }
            .searchable(text: $searchText, prompt: "Pick module")
            .preferredColorScheme(.dark)
    }
}

extension CGRect {
    var shaderArg: Shader.Argument {
        .float4(
            origin.x,
            origin.y,
            size.width,
            size.height
        )
    }
}

private let refDate = Date()

// Match the default search field fill color so rows blend with the search bar
private let tableBackgroundColor: SwiftUI.Color = Color(uiColor: .systemGray5)
private let textColor: SwiftUI.Color = Color.white
