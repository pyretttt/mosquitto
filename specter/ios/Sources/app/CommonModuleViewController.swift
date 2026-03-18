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
    
    struct ContextActions: OptionSet, Hashable, Sendable {
        var rawValue: Int
        
        init(rawValue: Int) {
            self.rawValue = rawValue
        }
        
        static let copyToBuffer = Self(rawValue: 1)
        static let saveToGallery = Self(rawValue: 1 << 1)
    }
    
    enum BottomBarControls: Hashable, Sendable {
        case contextMenu(ContextActions)
        case gallery
        case frontBackCameraSwitch
        case stack
        case shutter
        
        var order: Int {
            switch self {
            case .contextMenu: -2
            case .gallery: 2
            case .frontBackCameraSwitch: -1
            case .stack: 1
            case .shutter: 0
            }
        }
    }
    
    struct Config: Sendable {
        var options: [OptionModel]
        var bottomBarControls: [BottomBarControls]
    }

    private var topBarHost: UIHostingController<TopBarView>
    private var bottomBarHost: UIHostingController<CommonBottomBar>

    private let cameraModule: CameraStreamViewController
    private let config: Config

    init(
        cameraModule: CameraStreamViewController,
        config: Config
    ) {
        weak var weakSelf: CommonModuleViewController?
        self.cameraModule = cameraModule
        self.config = config
        topBarHost = modify(UIHostingController(
            rootView: TopBarView(
                onClose: { [cameraModule] in
                    switch cameraModule.cameraState.value {
                    case .notInitialized, .failedToInitialize, .streaming:
                        weakSelf?.dismiss(animated: true)
                    case .frozen:
                        cameraModule.inputActions.resetBufferContent()
                        let _ = cameraModule.inputActions.resumeStream()
                    }
                },
                onOptions: {
                    Task {
                        await cameraModule.inputActions.pauseStream().value
                        weakSelf?.presentOptionsPanel()
                    }
                }
            )
        )) {
            $0.view.backgroundColor = .clear
        }
        bottomBarHost = modify(UIHostingController(
            rootView: CommonBottomBar(
                controls: config.bottomBarControls,
                onShutter: {
                    weakSelf?.cameraModule.inputActions.didTapShutter(.photoTouchUpInside)
                },
                onOpenGallery: {
                    // TODO
                },
                onContextAction: { _ in
                    // TODO
                },
                onSwitchCamera: { _ in
                    // TODO
                },
                onStack: {
                    // TODO
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
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        let _ = cameraModule.inputActions.resumeStream()
    }
}

// MARK: - Bars

private extension CommonModuleViewController {
    private func presentOptionsPanel() {
        let optionsVC = modify(OptionsViewController(
            options: config.options,
            onDismiss: { [cameraModule] in
                let _ = cameraModule.inputActions.resumeStream()
            }
        )) {
            $0.modalPresentationStyle = .overFullScreen
        }
        present(optionsVC, animated: true)
    }

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

private struct TopBarView: View {
    var onClose: () -> Void
    var onOptions: () -> Void

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

            Button(action: onOptions) {
                Image(systemName: "gearshape")
                    .font(.system(size: 20, weight: .medium))
                    .padding(12)
                    .blendMode(.difference)
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 12)
        .frame(maxWidth: .infinity)
        .background(Color.clear)
    }
}

private struct CommonBottomBar: View {
    var controls: [CommonModuleViewController.BottomBarControls]
    var onShutter: () -> Void
    var onOpenGallery: () -> Void
    var onContextAction: (CommonModuleViewController.ContextActions) -> Void
    var onSwitchCamera: (Bool) -> Void
    var onStack: () -> Void

    @State private var isFrontCamera = false

    private var sorted: [CommonModuleViewController.BottomBarControls] {
        controls.sorted { $0.order < $1.order }
    }

    var body: some View {
        let left = sorted.filter { $0.order < 0 }
        let center = sorted.first { $0.order == 0 }
        let right = sorted.filter { $0.order > 0 }

        ZStack {
            if let center { controlView(center) }
            HStack {
                controlGroup(left)
                Spacer()
                controlGroup(right)
            }
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity)
        .background(Color.clear)
    }

    private func controlGroup(
        _ items: [CommonModuleViewController.BottomBarControls]
    ) -> some View {
        HStack(spacing: 8) {
            ForEach(items, id: \.self, content: controlView)
        }
    }

    @ViewBuilder
    private func controlView(
        _ control: CommonModuleViewController.BottomBarControls
    ) -> some View {
        switch control {
        case .contextMenu(let actions):
            contextMenuButton(actions: actions)
        case .gallery:
            iconButton("photo.on.rectangle", action: onOpenGallery)
        case .frontBackCameraSwitch:
            Button {
                isFrontCamera.toggle()
                onSwitchCamera(isFrontCamera)
            } label: {
                Image(systemName: "arrow.triangle.2.circlepath.camera")
                    .font(.system(size: bottomBarButtonSize, weight: .medium))
                    .foregroundStyle(isFrontCamera ? Color.yellow : .white)
                    .padding(12)
            }
            .buttonStyle(.plain)
        case .stack:
            iconButton("square.stack.3d.up", action: onStack)
        case .shutter:
            shutterButton
        }
    }

    private func contextMenuButton(
        actions: CommonModuleViewController.ContextActions
    ) -> some View {
        Menu {
            ForEach(actions.individualActions, id: \.rawValue) { action in
                Button(action: { onContextAction(action) }) {
                    Label(action.name, systemImage: action.icon)
                }
            }
        } label: {
            Image(systemName: "ellipsis.circle")
                .font(.system(size: bottomBarButtonSize, weight: .medium))
                .foregroundStyle(.white)
                .padding(12)
        }
    }

    private func iconButton(
        _ systemName: String,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: bottomBarButtonSize, weight: .medium))
                .foregroundStyle(.white)
                .padding(12)
        }
        .buttonStyle(.plain)
    }

    private var shutterButton: some View {
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
    }
}

extension CommonModuleViewController.ContextActions {
    fileprivate static let allKnown: [Self] = [.copyToBuffer, .saveToGallery]

    fileprivate var individualActions: [Self] {
        Self.allKnown.filter { contains($0) }
    }

    fileprivate var name: String {
        switch self {
        case .copyToBuffer: "Copy to buffer"
        case .saveToGallery: "Save to gallery"
        default: fatalError()
        }
    }

    fileprivate var icon: String {
        switch self {
        case .copyToBuffer: "doc.on.clipboard"
        case .saveToGallery: "square.and.arrow.down"
        default: fatalError()
        }
    }
}

private let bottomBarButtonSize: CGFloat = 24
