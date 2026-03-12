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
            var shutterState: ShutterKind?
        }
        
        case notInitialized
        case initializationFailure
        case streaming(Streaming)
        case frozen
        
        var streamingState: Streaming? {
            switch self {
            case .frozen, .notInitialized, .initializationFailure: nil
            case .streaming(let streaming): streaming
            }
        }
    }
    
    enum ShutterKind: Equatable {
        enum UIAction: Equatable {
            case touchDown
            case touchUpInside
        }
        case photoTouchUpInside
        case video(UIAction)
    }
    
    struct InputActions: Sendable {
        var didTapShutter: @Sendable @MainActor (ShutterKind) -> Void
        var pauseStream: @Sendable @MainActor () -> Void
        var resumeStream: @Sendable @MainActor () -> Void
        var replaceContent: @Sendable @MainActor () -> Void
    }
    
    struct OutputActions: Sendable {
        var didReceiveNewBuffer: @Sendable (CVImageBuffer) -> Void
        var didTakeAShot: @Sendable (CVImageBuffer) -> Void
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
            didTapShutter: { [weak self] shutterKind in
                guard let state = self?.cameraState.value.streamingState else { return }
                self?.cameraState.send(
                    .streaming(
                        modify(state) {
                            $0.shutterState = shutterKind
                        }
                    )
                )
            },
            pauseStream: { [captureSession] in
                captureSession.stopRunning()
            },
            resumeStream: { [captureSession] in
                captureSession.startIfNeeded()
            },
            replaceContent: {
                
            }
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
        captureSession.startIfNeeded()
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

    fileprivate func startIfNeeded() {
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
        startIfNeeded()
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
        Task { @MainActor in
            guard let streamingState = cameraState.value.streamingState else { return }
            switch streamingState.shutterState {
            case .photoTouchUpInside:
                cameraState.send(
                    .streaming(
                        modify(streamingState) {
                            $0.shutterState = nil
                        }
                    )
                )
            case .video(.touchDown):
                break
            case .video(.touchUpInside):
                break
            case .none:
                break
            }
            
            
        }
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
