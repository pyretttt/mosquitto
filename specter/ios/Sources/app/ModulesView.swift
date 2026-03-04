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
                            tableBackgroundColor
                                .opacity(0.9)
                        )
                }
                .listStyle(.insetGrouped)
                .navigationBarTitleDisplayMode(.inline)
                .scrollContentBackground(.hidden)
                .navigationTitle("Modules")
                .containerBackground(for: .navigation) {
                    TimelineView(.periodic(from: .now, by: 1.0 / 30.0)) { context in
                        let time = Float(refDate.timeIntervalSinceNow)
                        Rectangle()
                            .visualEffect { view, proxy in
                                view.colorEffect(
                                    ShaderLibrary.identity(
                                        proxy.frame(in: .global).shaderArg,
                                        .float(time)
                                    )
                                )
                            }
                    }
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

private let tableBackgroundColor: SwiftUI.Color = Color(hex: "#2B2534")
private let textColor: SwiftUI.Color = Color(hex: "#94919E")
