import SwiftUI
import AVFoundation
import UIKit   // For UIViewControllerRepresentable / UIImage

// MARK: - Backend response model

struct ProcessResponse: Codable {
    enum Intent: String, Codable {
        case describe, find, general
    }

    enum ActionTaken: String, Codable {
        case sceneDescription = "scene_description"
        case objectFoundNavigation = "object_found_navigation"
        case objectNotFound = "object_not_found"
        case generalQuery = "general_query"
    }

    struct NavigationMetrics: Codable {
        let distance: Double
        let direction: String
        let relativeAngle: Double

        enum CodingKeys: String, CodingKey {
            case distance
            case direction
            case relativeAngle = "relative_angle"
        }
    }

    struct Positions: Codable {
        struct Vec3: Codable {
            let x: Double
            let y: Double
            let z: Double
        }
        struct Orientation: Codable {
            let yaw: Double
            let pitch: Double
        }
        let targetPosition: Vec3
        let cameraPosition: Vec3
        let cameraOrientation: Orientation

        enum CodingKeys: String, CodingKey {
            case targetPosition = "target_position"
            case cameraPosition = "camera_position"
            case cameraOrientation = "camera_orientation"
        }
    }

    struct AdditionalData: Codable {
        let objectFound: Bool?
        let navigationMetrics: NavigationMetrics?
        let positions: Positions?

        enum CodingKeys: String, CodingKey {
            case objectFound = "object_found"
            case navigationMetrics = "navigation_metrics"
            case positions
        }
    }

    let intent: Intent
    let transcript: String
    let targetObject: String
    let actionTaken: ActionTaken
    let responseText: String
    let audioOutput: String
    let audioBase64: String?      // base64-encoded audio
    let audioFormat: String?      // audio format, e.g. "mp3" or "wav"
    let additionalData: AdditionalData?

    enum CodingKeys: String, CodingKey {
        case intent
        case transcript
        case targetObject = "target_object"
        case actionTaken = "action_taken"
        case responseText = "response_text"
        case audioOutput = "audio_output"
        case audioBase64 = "audio_base64"
        case audioFormat = "audio_format"
        case additionalData = "additional_data"
    }
}

// MARK: - Audio recorder

final class AudioRecorder: NSObject, ObservableObject {
    @Published var isRecording = false

    private var recorder: AVAudioRecorder?

    func startRecording() throws -> URL {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
        try session.setActive(true)

        session.requestRecordPermission { granted in
            if !granted {
                print("Microphone permission not granted")
            }
        }

        let url = Self.recordingURL()

        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: 16_000,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
        ]

        recorder = try AVAudioRecorder(url: url, settings: settings)
        recorder?.prepareToRecord()
        recorder?.record()
        isRecording = true

        return url
    }

    func stopRecording() -> URL? {
        recorder?.stop()
        isRecording = false
        return recorder?.url
    }

    static func recordingURL() -> URL {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return documents.appendingPathComponent("user_query.wav")
    }
}

// MARK: - Auto-capture camera (takes a photo as soon as it opens)

struct AutoCaptureCameraView: UIViewControllerRepresentable {
    @Environment(\.dismiss) private var dismiss
    let onCaptured: (UIImage) -> Void

    func makeUIViewController(context: Context) -> AutoCaptureCameraViewController {
        let vc = AutoCaptureCameraViewController()
        vc.onCaptured = { image in
            onCaptured(image)
            dismiss()
        }
        return vc
    }

    func updateUIViewController(_ uiViewController: AutoCaptureCameraViewController, context: Context) {}
}

final class AutoCaptureCameraViewController: UIViewController, AVCapturePhotoCaptureDelegate {
    private let session = AVCaptureSession()
    private let output = AVCapturePhotoOutput()
    private var previewLayer: AVCaptureVideoPreviewLayer?
    var onCaptured: ((UIImage) -> Void)?

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        configureSession()
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)

        DispatchQueue.global(qos: .userInitiated).async {
            self.session.startRunning()

            // Wait a bit for focus/exposure, then auto-capture
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
                let settings = AVCapturePhotoSettings()
                self.output.capturePhoto(with: settings, delegate: self)
            }
        }
    }

    private func configureSession() {
        session.beginConfiguration()
        session.sessionPreset = .photo

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                   for: .video,
                                                   position: .back) else {
            print("No back camera")
            session.commitConfiguration()
            return
        }

        do {
            let input = try AVCaptureDeviceInput(device: device)
            if session.canAddInput(input) {
                session.addInput(input)
            }

            if session.canAddOutput(output) {
                session.addOutput(output)
            }

            session.commitConfiguration()

            let layer = AVCaptureVideoPreviewLayer(session: session)
            layer.videoGravity = .resizeAspectFill
            layer.frame = view.bounds
            view.layer.addSublayer(layer)
            previewLayer = layer
        } catch {
            print("Error configuring camera: \(error)")
            session.commitConfiguration()
        }
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = view.bounds
    }

    func photoOutput(_ output: AVCapturePhotoOutput,
                     didFinishProcessingPhoto photo: AVCapturePhoto,
                     error: Error?) {
        if let error = error {
            print("Capture error: \(error)")
            return
        }

        guard let data = photo.fileDataRepresentation(),
              let image = UIImage(data: data) else {
            print("Failed to get image data")
            return
        }

        onCaptured?(image)
    }

    deinit {
        if session.isRunning {
            session.stopRunning()
        }
    }
}

// MARK: - ViewModel: networking + audio playback state

@MainActor
final class ProcessViewModel: NSObject, ObservableObject, AVAudioPlayerDelegate {
    // Your backend URL
    private let baseURL = URL(string: "http://131.159.220.251:8000")!

    @Published var audioURL: URL?
    @Published var capturedImage: UIImage?
    @Published var isSending = false
    @Published var errorMessage: String?
    @Published var result: ProcessResponse?
    @Published var isPlaying = false      // whether audio is currently playing

    private var player: AVAudioPlayer?

    func sendRequest(maxSearchDuration: Int? = 5) async {
        errorMessage = nil
        result = nil

        guard let audioURL = audioURL else {
            errorMessage = "No recording available yet."
            return
        }
        guard let image = capturedImage else {
            errorMessage = "No photo captured yet."
            return
        }

        do {
            let audioData = try Data(contentsOf: audioURL)

            // Resize and compress image before sending
            let resizedImage = image.resized(maxDimension: 1024)
            guard let imageData = resizedImage.jpegData(compressionQuality: 0.6) else {
                errorMessage = "Failed to encode image."
                return
            }

            isSending = true
            defer { isSending = false }

            let response = try await upload(
                audioData: audioData,
                imageData: imageData,
                maxSearchDuration: maxSearchDuration
            )
            self.result = response

            // Prefer base64 audio
            try await playBackendAudio(from: response)
        } catch {
            errorMessage = "Request failed: \(error.localizedDescription)"
        }
    }

    private func upload(audioData: Data,
                        imageData: Data,
                        maxSearchDuration: Int?) async throws -> ProcessResponse {

        var request = URLRequest(url: baseURL.appendingPathComponent("process"))
        request.httpMethod = "POST"

        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()

        body.appendMultipartField(
            name: "audio",
            filename: "user_query.wav",
            mimeType: "audio/wav",
            data: audioData,
            boundary: boundary
        )

        body.appendMultipartField(
            name: "frames",
            filename: "frame_0.jpg",
            mimeType: "image/jpeg",
            data: imageData,
            boundary: boundary
        )

        if let maxSearchDuration {
            body.appendTextField(
                name: "max_search_duration",
                value: String(maxSearchDuration),
                boundary: boundary
            )
        }

        body.append("--\(boundary)--\r\n".data(using: .utf8)!)

        request.httpBody = body

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResp = response as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }

        guard (200..<300).contains(httpResp.statusCode) else {
            let text = String(data: data, encoding: .utf8) ?? "<no body>"
            throw NSError(domain: "BackendError", code: httpResp.statusCode, userInfo: [
                NSLocalizedDescriptionKey: "HTTP \(httpResp.statusCode): \(text)"
            ])
        }

        let decoder = JSONDecoder()
        let result = try decoder.decode(ProcessResponse.self, from: data)
        return result
    }

    // Prefer playing base64 audio
    private func playBackendAudio(from response: ProcessResponse) async throws {
        // Reset any previous player
        player?.stop()
        isPlaying = false

        if let base64String = response.audioBase64, !base64String.isEmpty {
            guard let audioData = Data(base64Encoded: base64String) else {
                throw NSError(domain: "AudioError", code: -1, userInfo: [
                    NSLocalizedDescriptionKey: "Failed to decode base64 audio data."
                ])
            }

            let newPlayer = try AVAudioPlayer(data: audioData)
            newPlayer.delegate = self
            newPlayer.prepareToPlay()
            newPlayer.play()

            player = newPlayer
            isPlaying = true
        } else {
            // Fallback: try downloading from audio_output URL
            print("Warning: No audio_base64 found, trying audioOutput URL")
            let audioPath = response.audioOutput
            let url: URL
            if audioPath.lowercased().hasPrefix("http") {
                url = URL(string: audioPath)!
            } else {
                let trimmed = audioPath.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
                url = baseURL.appendingPathComponent(trimmed)
            }

            let (data, _) = try await URLSession.shared.data(from: url)
            let newPlayer = try AVAudioPlayer(data: data)
            newPlayer.delegate = self
            newPlayer.prepareToPlay()
            newPlayer.play()

            player = newPlayer
            isPlaying = true
        }
    }

    // AVAudioPlayerDelegate
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        isPlaying = false
    }

    func audioPlayerDecodeErrorDidOccur(_ player: AVAudioPlayer, error: Error?) {
        isPlaying = false
        if let error = error {
            print("Audio decode error: \(error)")
        }
    }
}

// MARK: - multipart/form-data helpers

private extension Data {
    mutating func appendMultipartField(name: String,
                                       filename: String,
                                       mimeType: String,
                                       data: Data,
                                       boundary: String) {
        append("--\(boundary)\r\n".data(using: .utf8)!)
        append("Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(filename)\"\r\n".data(using: .utf8)!)
        append("Content-Type: \(mimeType)\r\n\r\n".data(using: .utf8)!)
        append(data)
        append("\r\n".data(using: .utf8)!)
    }

    mutating func appendTextField(name: String,
                                  value: String,
                                  boundary: String) {
        append("--\(boundary)\r\n".data(using: .utf8)!)
        append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n".data(using: .utf8)!)
        append("\(value)\r\n".data(using: .utf8)!)
    }
}

// MARK: - Main UI: round button + auto-capture camera + pulsing while playing

struct ContentView: View {
    @StateObject private var recorder = AudioRecorder()
    @StateObject private var viewModel = ProcessViewModel()

    @State private var showCamera = false
    @State private var isPulsing = false   // drives the pulsing animation

    var body: some View {
        NavigationStack {
            VStack(spacing: 24) {
                Spacer()

                // Center round button
                Button {
                    onMainButtonTapped()
                } label: {
                    ZStack {
                        Circle()
                            .fill(recorder.isRecording ? Color.red : Color.blue)
                            .frame(width: 120, height: 120)
                            .scaleEffect(
                                viewModel.isPlaying
                                ? (isPulsing ? 1.15 : 0.95)   // pulse while playing
                                : 1.0
                            )

                        Image(systemName: recorder.isRecording ? "stop.fill" : "mic.fill")
                            .font(.system(size: 40))
                            .foregroundColor(.white)
                    }
                }
                .disabled(viewModel.isSending || showCamera)
                .accessibilityLabel(recorder.isRecording ? "Stop recording" : "Start recording")
                .accessibilityHint("Double-tap to start or end a voice query")

                if recorder.isRecording {
                    Text("Recording…")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                } else if viewModel.isPlaying {
                    Text("Playing response…")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                } else {
                    Text("Tap the round button to start recording.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }

                if viewModel.isSending {
                    HStack(spacing: 8) {
                        ProgressView()
                        Text("Generating...")
                    }
                }

                Spacer()

                if let error = viewModel.errorMessage {
                    Text(error)
                        .foregroundStyle(.red)
                        .font(.footnote)
                        .multilineTextAlignment(.leading)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .padding()
            .navigationTitle("The Third Eye")
            .sheet(isPresented: $showCamera) {
                // Auto-capture camera: takes a photo and returns it
                AutoCaptureCameraView { image in
                    viewModel.capturedImage = image
                    Task {
                        await viewModel.sendRequest(maxSearchDuration: 5)
                    }
                }
            }
        }
        .onChange(of: viewModel.isPlaying) { playing in
            if playing {
                // Start pulsing: toggle once, animate with repeatForever
                isPulsing = false
                withAnimation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true)) {
                    isPulsing = true
                }
            } else {
                // Stop pulsing smoothly
                withAnimation(.easeOut(duration: 0.2)) {
                    isPulsing = false
                }
            }
        }
    }

    private func onMainButtonTapped() {
        // Ignore input while sending or camera is presented to avoid weird toggles
        if viewModel.isSending || showCamera {
            return
        }

        if recorder.isRecording {
            // Stop recording → open camera
            let url = recorder.stopRecording()
            viewModel.audioURL = url
            showCamera = true
        } else {
            // Start recording
            do {
                let url = try recorder.startRecording()
                viewModel.audioURL = url
            } catch {
                print("Failed to start recording: \(error)")
                viewModel.errorMessage = "Failed to start recording: \(error.localizedDescription)"
            }
        }
    }
}

// MARK: - UIImage resize helper

extension UIImage {
    /// Resize so that the longest side is maxDimension (in pixels)
    func resized(maxDimension: CGFloat) -> UIImage {
        let maxCurrent = max(size.width, size.height)
        // Already small enough
        guard maxCurrent > maxDimension else { return self }

        let scale = maxDimension / maxCurrent
        let newSize = CGSize(width: size.width * scale, height: size.height * scale)

        UIGraphicsBeginImageContextWithOptions(newSize, true, 1.0)
        defer { UIGraphicsEndImageContext() }

        self.draw(in: CGRect(origin: .zero, size: newSize))
        return UIGraphicsGetImageFromCurrentImageContext() ?? self
    }
}
