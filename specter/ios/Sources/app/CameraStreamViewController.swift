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
import CoreImage

@MainActor
final class CameraStreamViewController: UIViewController {
    enum State: Equatable {
        struct Streaming: Equatable {
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
        // case video(UIAction)
    }
    
    struct InputActions: Sendable {
        var didTapShutter: @Sendable @MainActor (ShutterKind) -> Void
        var pauseStream: @Sendable @MainActor () -> Void
        var resumeStream: @Sendable @MainActor () -> Void
        var replaceContent: @Sendable @MainActor (CVImageBuffer) -> Void
    }
    
    struct OutputActions: Sendable {
        var didReceiveNewBuffer: @Sendable (CVImageBuffer) -> Void
        var didTakeAShot: @Sendable @MainActor (CVImageBuffer) -> Void
    }

    private let captureSession = AVCaptureSession()
    private var previewLayer: AVCaptureVideoPreviewLayer
    private let closeButton = UIButton(configuration: .plain())
    private let videoOutputQueue = DispatchQueue(
        label: "camera.video.output.queue",
        qos: .userInitiated
    )
    private let frozenImageView = modify(UIImageView()) {
        $0.contentMode = .scaleAspectFill
        $0.clipsToBounds = true
        $0.alpha = 0
    }
    
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
                captureSession.stopRunningDetached()
            },
            resumeStream: { [captureSession] in
                captureSession.startIfNeeded()
            },
            replaceContent: { [weak self, captureSession] imageBuffer in
                captureSession.stopRunningDetached()
                self?.cameraState.send(.frozen)
                guard let uiImage = imageBuffer.ciImage?.cgImage?.uiImage else { return }
                self?.showFrozenImage(uiImage)
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
        frozenImageView.frame = view.bounds
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        Task {
            await captureSession.startIfNeeded()?.value
            cameraState.send(.streaming(State.Streaming()))
        }
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        captureSession.stopRunningDetached()
        cameraState.send(.frozen)
    }
}

private extension CameraStreamViewController {
    func setupUI() {
        view.layer.insertSublayer(previewLayer, at: 0)
        view.addSubview(frozenImageView)
    }

    func showFrozenImage(_ image: UIImage) {
        frozenImageView.image = image
        frozenImageView.alpha = 0
        frozenImageView.transform = CGAffineTransform(scaleX: 1.05, y: 1.05)

        let flashView = modify(UIView()) {
            $0.backgroundColor = .white
            $0.frame = view.bounds
            $0.alpha = 0
        }
        view.addSubview(flashView)

        UIView.animate(withDuration: 0.08) {
            flashView.alpha = 1
        } completion: { _ in
            UIView.animate(withDuration: 0.35, delay: 0, usingSpringWithDamping: 0.85, initialSpringVelocity: 0.5) {
                flashView.alpha = 0
                self.frozenImageView.alpha = 1
                self.frozenImageView.transform = .identity
            } completion: { _ in
                flashView.removeFromSuperview()
            }
        }
    }
}

extension AVCaptureSession {
    @discardableResult
    fileprivate func stopRunningDetached() -> Task<Void, Never> {
        Task.detached {
            self.stopRunning()
        }
    }
    
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

    @discardableResult
    fileprivate func startIfNeeded() -> Task<Void, Never>? {
        if isRunning { return nil }
        return Task(operation: startRunning)
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
            if let connection = output.connection(with: .video) {
                if connection.isVideoRotationAngleSupported(90) {
                    connection.videoRotationAngle = 90
                }
            }
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
                outputActions.didTakeAShot(imageBuffer)
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

extension CVImageBuffer {
    var ciImage: CIImage? {
        CIImage(cvImageBuffer: self)
    }
}

extension CIImage {
    var cgImage: CGImage? {
        ciContext.createCGImage(self, from: extent)
    }
}

extension CGImage {
    var uiImage: UIImage? {
        UIImage(cgImage: self)
    }
}

private let ciContext = CIContext()
