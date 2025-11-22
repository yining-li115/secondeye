import Foundation
import AVFAudio

final class VolumeButtonObserver: NSObject, ObservableObject {
    private let audioSession = AVAudioSession.sharedInstance()
    private var observation: NSKeyValueObservation?

    private var lastVolume: Float = 0.5
    private var isInitial = true
    private let debounceInterval: TimeInterval = 0.8   // 两次按键至少间隔 0.8 秒
    private var lastTriggerDate: Date?

    /// 每次识别到“一次有效的音量键按下”时回调
    var onVolumeButtonPressed: (() -> Void)?

    override init() {
        super.init()
        setup()
    }

    private func setup() {
        do {
            try audioSession.setActive(true, options: [])
            lastVolume = audioSession.outputVolume
        } catch {
            print("Failed to activate audio session for volume observing: \(error)")
        }

        observation = audioSession.observe(\.outputVolume, options: [.new]) { [weak self] _, change in
            guard let self = self else { return }
            guard let newVolume = change.newValue else { return }

            // 第一次回调是当前音量，不算按键
            if self.isInitial {
                self.isInitial = false
                self.lastVolume = newVolume
                return
            }

            // 很小的变化忽略（抖动）
            if abs(newVolume - self.lastVolume) < 0.02 {
                self.lastVolume = newVolume
                return
            }

            let now = Date()
            if let last = self.lastTriggerDate,
               now.timeIntervalSince(last) < self.debounceInterval {
                // 防抖期间的后续变化只当系统连续调音量，不算多次按键
                self.lastVolume = newVolume
                return
            }

            self.lastVolume = newVolume
            self.lastTriggerDate = now

            DispatchQueue.main.async {
                self.onVolumeButtonPressed?()
            }
        }
    }

    deinit {
        observation?.invalidate()
    }
}
