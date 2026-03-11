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
import Combine

@MainActor
final class CameraStreamViewController: UIViewController {
    enum State: Equatable {
        struct Streaming: Equatable {
            var frameBuffer: CMSampleBuffer?
        }
        
        case notInitialized
        case initializationFailure
        case streaming(Streaming)
        case frozen
    }
    
    struct InputActions: Sendable {
        var didTapShutter: @Sendable () -> Void
        var pauseStream: @Sendable () -> Void
        var replaceContent: @Sendable () -> Void
    }
    
    struct OutputActions: Sendable {
        var didReceiveNewBuffer: @Sendable (CVImageBuffer) -> Void
    }

    private let captureSession = AVCaptureSession()
    private var previewLayer: AVCaptureVideoPreviewLayer
    private let closeButton = UIButton(configuration: .plain())
    private let videoOutputQueue = DispatchQueue(
        label: "camera.video.output.queue",
        qos: .userInitiated
    )
    
    private let outputActions: OutputActions
    
    
    var inputActions: InputActions {
        InputActions(
            didTapShutter: {},
            pauseStream: {},
            replaceContent: {}
        )
    }
    
    let cameraState = CurrentValueSubject<State, Never>(.notInitialized)

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
        captureSession.setupDeviceWithAccess(
            sampleBufferDelegate: self,
            queue: videoOutputQueue
        ) {
            self.cameraState.send(.initializationFailure)
            self.showAlert("Failed to setup camera configuration")
        }
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer.frame = view.bounds
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        captureSession.startSessionIfNeeded()
        cameraState.send(.streaming(State.Streaming(frameBuffer: nil)))
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        captureSession.stopRunning()
        cameraState.send(.frozen)
    }
}


private extension CameraStreamViewController {
    func setupUI() {
        view.layer.insertSublayer(previewLayer, at: 0)
    }
}

extension AVCaptureSession {
    fileprivate func setupDeviceWithAccess(
        sampleBufferDelegate: AVCaptureVideoDataOutputSampleBufferDelegate,
        queue: DispatchQueue,
        onError: @escaping () -> Void
    ) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            configureSession(sampleBufferDelegate: sampleBufferDelegate, queue: queue, onError: onError)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                Task { @MainActor in
                    granted
                        ? self?.configureSession(
                            sampleBufferDelegate: sampleBufferDelegate,
                            queue: queue,
                            onError: onError
                        )
                        : onError()
                }
            }
        case .restricted, .denied:
            onError()
        @unknown default:
            assertionFailure("Unknown case")
        }
    }

    
    fileprivate func startSessionIfNeeded() {
        if isRunning { return }
        Task(operation: startRunning)
    }

    fileprivate func configureSession(
        sampleBufferDelegate: AVCaptureVideoDataOutputSampleBufferDelegate,
        queue: DispatchQueue,
        onError: () -> Void
    ) {
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
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.setSampleBufferDelegate(sampleBufferDelegate, queue: queue)
        if canAddOutput(output) {
            addOutput(output)
        }

        commitConfiguration()
        startSessionIfNeeded()
    }

}

extension CameraStreamViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        outputActions.didReceiveNewBuffer(imageBuffer)
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
