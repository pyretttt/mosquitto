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
    enum BottomPanel {
        case shutter
    }

    private let bottomPanel: BottomPanel
    private let captureSession = AVCaptureSession()
    private var previewLayer: AVCaptureVideoPreviewLayer
    private let closeButton = UIButton(configuration: .plain())
    private let bottomContainer = UIView()
    private var shutterButton: UIButton = UIButton(type: .custom)
    private var permissionDeniedLabel: UILabel?

    init(bottomPanel: BottomPanel = .shutter) {
        self.bottomPanel = bottomPanel
        previewLayer = modify(AVCaptureVideoPreviewLayer(session: captureSession)) {
            $0.videoGravity = .resizeAspectFill
        }
        super.init(nibName: nil, bundle: nil)
        modalPresentationStyle = .fullScreen
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
        
        modify(closeButton) {
            $0.setImage(UIImage(systemName: "xmark"), for: .normal)
            $0.tintColor = .white
            $0.backgroundColor = UIColor.black.withAlphaComponent(0.6)
            $0.layer.cornerRadius = 16
            $0.configuration?.contentInsets = NSDirectionalEdgeInsets(top: 8, leading: 8, bottom: 8, trailing: 8)
            $0.addTarget(self, action: #selector(didTapClose), for: .touchUpInside)
        }
        view.addSubview(closeButton)
        closeButton.snp.makeConstraints { make in
            make.top.equalTo(view.safeAreaLayoutGuide.snp.top).offset(12)
            make.leading.equalTo(view.safeAreaLayoutGuide.snp.leading).offset(12)
        }
        
        bottomContainer.backgroundColor = UIColor.black.withAlphaComponent(0.4)
        view.addSubview(bottomContainer)

        bottomContainer.snp.makeConstraints { make in
            make.leading.trailing.equalTo(view)
            make.bottom.equalTo(view.safeAreaLayoutGuide)
            make.height.equalTo(120)
        }

        switch bottomPanel {
        case .shutter:
            _ = modify(shutterButton) {
                $0.backgroundColor = .white
                $0.layer.cornerRadius = 34
                $0.layer.borderColor = UIColor.lightGray.cgColor
                $0.layer.borderWidth = 3
                $0.addTarget(self, action: #selector(didTapShutter), for: .touchUpInside)
                bottomContainer.addSubview($0)
                $0.snp.makeConstraints { make in
                    make.centerX.centerY.equalTo(bottomContainer)
                    make.width.height.equalTo(68)
                }
            }
        }
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

// MARK: - Actions
private extension CameraStreamViewController {
    @objc func didTapClose() {
        dismiss(animated: true)
    }

    @objc func didTapShutter(_ sender: UIButton) {
        // Default behavior is to take a simple flash animation
        UIView.animate(withDuration: 0.1, animations: {
            self.view.backgroundColor = .white
        }, completion: { _ in
            UIView.animate(withDuration: 0.25) {
                self.view.backgroundColor = .black
            }
        })
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
