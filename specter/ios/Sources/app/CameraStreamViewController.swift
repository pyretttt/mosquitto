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
final class CameraStreamViewController: UIViewController, Sendable {
    enum State: Equatable {
        struct Streaming: Equatable {
            var shutterState: ShutterKind?
            
            static let empty = Streaming()
        }

        case notInitialized
        case failedToInitialize
        case streaming(Streaming)
        case frozen

        var streamingState: Streaming? {
            switch self {
            case .frozen, .notInitialized, .failedToInitialize: nil
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
        var pauseStream: @Sendable @MainActor () -> Task<Void, Never>
        var resumeStream: @Sendable @MainActor () -> Task<Void, Never>
        var setBufferContent: @Sendable @MainActor (CVImageBuffer) -> Void
        var resetBufferContent: @Sendable @MainActor () -> Void
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
            pauseStream: { [weak self, captureSession] in
                return Task {
                    await captureSession.stopRunningDetached().value
                    self?.cameraState.send(.frozen)
                }
            },
            resumeStream: { [weak self, captureSession] in
                return Task {
                    await captureSession.startIfNeeded()?.value
                    self?.cameraState.send(.streaming(.empty))
                }
            },
            setBufferContent: { [weak self, captureSession] imageBuffer in
                captureSession.stopRunningDetached()
                self?.cameraState.send(.frozen)
                guard let uiImage = imageBuffer.ciImage?.cgImage?.uiImage else { return }
                self?.toggleFrozenContent(visible: true, image: uiImage)
            },
            resetBufferContent: { [weak self] in
                self?.toggleFrozenContent(visible: false, image: nil)
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
            self.cameraState.send(.failedToInitialize)
            self.showAlert("Failed to setup camera configuration")
        }
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer.frame = view.bounds
        frozenImageView.frame = view.bounds
    }
}

private extension CameraStreamViewController {
    func setupUI() {
        view.layer.insertSublayer(previewLayer, at: 0)
        view.addSubview(frozenImageView)
    }

    func toggleFrozenContent(visible: Bool, image: UIImage?) {
        frozenImageView.image = image
        frozenImageView.alpha = visible ? 0.0 : 1.0
        frozenImageView.transform = CGAffineTransform(scaleX: 1.05, y: 1.05)

        UIView.animate(withDuration: 0.25) {
            self.frozenImageView.alpha = visible ? 1.0 : 0.0
            self.frozenImageView.transform = .identity
        }
    }
}

extension AVCaptureSession {
    @discardableResult
    fileprivate func stopRunningDetached() -> Task<Void, Never> {
        Task.detached(priority: .userInitiated) {
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
    var uiImage: UIImage? {
        UIImage(ciImage: self)
    }

    var cgImage: CGImage? {
        ciContext.createCGImage(self, from: extent)
    }
}

extension CGImage {
    var uiImage: UIImage? {
        UIImage(cgImage: self)
    }
}

private let ciContext = CIContext(mtlDevice: MTLCreateSystemDefaultDevice()!)
