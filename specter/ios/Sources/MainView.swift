//
//  MainView.swift
//  app
//
//  Created by Бакулин Семен Александрович on 21.02.2026.
//

import Foundation
import SwiftUI

struct MainView: View {
    var body: some View {
        VStack(spacing: 12) {
            Text("Hello from MainView")
                .font(.title2)
                .fontWeight(.semibold)
            Text("This SwiftUI view is hosted inside ViewController.")
                .font(.body)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
        }
        .padding()
    }
}
