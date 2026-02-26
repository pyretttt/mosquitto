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
                        .onTapGesture {
                            moduleService.didTapModule(item)
                        }
                        .listRowBackground(Color.clear)
                }
                .listStyle(.plain)
                .navigationBarTitleDisplayMode(.inline)
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
