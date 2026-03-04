//
//  CameraStreamViewController.swift
//  app
//
//  Created by Бакулин Семен Александрович on 04.03.2026.
//

import UIKit
import AVFoundation
import SnapKit
import ios_Base

@MainActor
final class CameraStreamViewController: UIViewController {
    struct InputActions {
        var didTapShutter: () -> Void
    }
    
    struct OutputActions {
        var didReceiveNewBuffer: () -> Void
    }

    private let captureSession = AVCaptureSession()
    private var previewLayer: AVCaptureVideoPreviewLayer
    private let closeButton = UIButton(configuration: .plain())
    
    private let outputActions: OutputActions
    
    
    var inputActions: InputActions {
        InputActions(
            didTapShutter: {},
        )
    }

    init(
        outputActions: OutputActions
    ) {
        self.outputActions = outputActions
        previewLayer = modify(AVCaptureVideoPreviewLayer(session: captureSession)) {
            $0.videoGravity = .resizeAspectFill
        }
        super.init(nibName: nil, bundle: nil)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        setupUI()
        setupDeviceWithAccess()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer.frame = view.bounds
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        captureSession.startSessionIfNeeded()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        captureSession.stopRunning()
    }
}

// MARK: - UI setup
private extension CameraStreamViewController {
    func setupUI() {
        view.layer.insertSublayer(previewLayer, at: 0)
    }
}

// MARK: - Camera handling
private extension CameraStreamViewController {
    func setupDeviceWithAccess() {
        let onError = { self.showAlert("Failed to setup camera configuration") }
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            captureSession.configureSession(onError: onError)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                Task { @MainActor in
                    granted
                        ? self?.captureSession.configureSession(onError: onError)
                        : self?.showAlert("Camera permission is not granted")
                }
            }
        case .restricted, .denied:
            showAlert("Camera permission denied, grant access in settings")
        @unknown default:
            assertionFailure("Unknown case")
        }
    }
}

extension AVCaptureSession {
    fileprivate func startSessionIfNeeded() {
        if isRunning { return }
        Task(operation: startRunning)
    }

    fileprivate func configureSession(onError: () -> Void) {
        beginConfiguration()
        sessionPreset = .high

        if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
           let input = try? AVCaptureDeviceInput(device: device),
           canAddInput(input) {
            addInput(input)
        } else {
            commitConfiguration()
            return onError()
        }

        let output = AVCaptureVideoDataOutput()
        if canAddOutput(output) {
            addOutput(output)
        }

        commitConfiguration()
        startSessionIfNeeded()
    }

}

extension UIViewController {
    func showAlert(_ text: String) {
        let alert = modify(UIAlertController(title: nil, message: text, preferredStyle: .alert)) {
            $0.addAction(UIAlertAction(title: "OK", style: .default, handler: nil))
        }
        present(alert, animated: true)
    }
}
