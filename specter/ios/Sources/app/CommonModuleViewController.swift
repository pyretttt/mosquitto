//
//  CommonModuleViewController.swift
//  app
//
//  Created by Codex on 04.03.2026.
//

import UIKit
import SwiftUI
import ios_Base
import SnapKit

@MainActor
final class CommonModuleViewController: PassThroughViewController {

    private var topBarHost: UIHostingController<TopBarView>
    private var bottomBarHost: UIHostingController<CommonBottomBar>

    private let cameraModule: CameraStreamViewController

    init(cameraModule: CameraStreamViewController) {
        weak var weakSelf: CommonModuleViewController?
        self.cameraModule = cameraModule
        topBarHost = modify(UIHostingController(
            rootView: TopBarView { [cameraModule] in
                switch cameraModule.cameraState.value {
                case .notInitialized, .failedToInitialize, .streaming:
                    weakSelf?.dismiss(animated: true)
                case .frozen:
                    cameraModule.resetBufferContent()
                    cameraModule.resumeStream()
                }
            }
        )) {
            $0.view.backgroundColor = .clear
        }
        bottomBarHost = modify(UIHostingController(
            rootView: CommonBottomBar(
                onShutter: {
                    weakSelf?.cameraModule.inputActions.didTapShutter(.photoTouchUpInside)
                },
                onOpenGallery: {
                    weakSelf?.onOpenGallery?()
                }
            )
        )) {
            $0.view.backgroundColor = .clear
        }

        super.init(nibName: nil, bundle: nil)
        weakSelf = self
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .clear
        setupUI()
    }
}

// MARK: - Bars

private extension CommonModuleViewController {
    private func setupUI() {
        addChild(cameraModule)
        view.addSubview(cameraModule.view)
        cameraModule.view.snp.makeConstraints {
            $0.edges.equalToSuperview()
        }

        addChild(topBarHost)
        view.addSubview(topBarHost.view)
        topBarHost.view.snp.makeConstraints {
            $0.leading.trailing.equalToSuperview()
            $0.top.equalTo(view.safeAreaLayoutGuide.snp.top)
            $0.height.equalTo(44)
        }
        addChild(bottomBarHost)
        view.addSubview(bottomBarHost.view)
        bottomBarHost.view.snp.makeConstraints {
            $0.bottom.leading.trailing.equalToSuperview()
            $0.height.equalTo(80)
        }
    }
}

// MARK: - SwiftUI components

/// Transparent top bar with a single close button on the leading edge.
private struct TopBarView: View {
    var onClose: () -> Void

    var body: some View {
        HStack {
            Button(action: onClose) {
                Image(systemName: "xmark")
                    .font(.system(size: 22, weight: .semibold))
                    .padding(12)
                    .blendMode(.difference)
            }
            .buttonStyle(.plain)

            Spacer()
        }
        .padding(.horizontal, 12)
        .frame(maxWidth: .infinity)
        .background(Color.clear)
    }
}

/// Transparent bottom bar with shutter centered and gallery button alongside.
private struct CommonBottomBar: View {
    var onShutter: () -> Void
    var onOpenGallery: () -> Void

    var body: some View {
        HStack {
            Button(action: onOpenGallery) {
                Image(systemName: "photo.on.rectangle")
                    .font(.system(size: 20, weight: .medium))
                    .padding(12)
            }
            .buttonStyle(.plain)

            Spacer()

            Button(action: onShutter) {
                ZStack {
                    Circle()
                        .stroke(Color.white.opacity(0.8), lineWidth: 3)
                        .frame(width: 64, height: 64)
                    Circle()
                        .fill(Color.white.opacity(0.9))
                        .frame(width: 50, height: 50)
                }
            }
            .buttonStyle(.plain)

            Spacer()

            // Spacer to keep shutter centered even with two buttons; could host future actions.
            Color.clear
                .frame(width: 44, height: 44)
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity)
        .background(Color.clear)
    }
}
