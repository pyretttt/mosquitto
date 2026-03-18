//
//  ImageProcessingToolSelectionView.swift
//  app
//
//  Created by Бакулин Семен Александрович on 18.03.2026.
//

import SwiftUI

struct ImageProcessingToolSelectionView: View {
    var tools: [String]
    var onClose: () -> Void
    var onSelectTool: (Int) -> Void

    var body: some View {
        NavigationStack {
            List(tools.indices, id: \.self) { index in
                Button(action: { onSelectTool(index) }) {
                    HStack {
                        Text(tools[index])
                            .font(.system(size: 18, weight: .medium))
                            .foregroundStyle(Color.white)
                        Spacer()
                        Image(systemName: "chevron.right")
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundStyle(Color.white.opacity(0.6))
                    }
                    .padding(.vertical, 8)
                }
                .buttonStyle(.plain)
                .listRowSeparator(.hidden)
                .listRowBackground(toolRowColor)
            }
            .listStyle(.insetGrouped)
            .scrollContentBackground(.hidden)
            .navigationTitle("Tools")
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Close", action: onClose)
                }
            }
            .background(toolScreenColor.ignoresSafeArea())
        }
        .preferredColorScheme(.dark)
    }
}

private let toolScreenColor = Color.black
private let toolRowColor = Color.white.opacity(0.08)
