import Foundation
import MLX

#if canImport(AVFoundation)
import AVFoundation
#endif

public enum AudioReader {
    public enum AudioReaderError: LocalizedError {
        case avFoundationUnavailable
        case unsupportedFormat(URL)
        case decodeFailed(String)
        case empty(URL)

        public var errorDescription: String? {
            switch self {
            case .avFoundationUnavailable:
                return "AVFoundation is not available on this platform"
            case .unsupportedFormat(let url):
                return "Unsupported audio format at \(url.path)"
            case .decodeFailed(let detail):
                return "Failed to decode audio: \(detail)"
            case .empty(let url):
                return "Audio file at \(url.path) contains no samples"
            }
        }
    }

    /// Loads an audio file from disk and returns it as an MLXArray of shape
    /// `[1, 2, samples]`, float32 in the range `[-1, 1]`, resampled to 44.1 kHz
    /// and downmixed/upmixed to stereo as needed.
    public static func loadStereo44k(url: URL) throws -> MLXArray {
        let raw = try read(url: url)
        return try loadStereo44k(samples: raw.samples, sampleRate: raw.sampleRate, channelCount: raw.channelCount)
    }

    /// Same as `loadStereo44k(url:)` but for already-decoded interleaved PCM.
    /// `samples.count` must equal `frameCount * channelCount`.
    public static func loadStereo44k(samples: [Float], sampleRate: Int, channelCount: Int) throws -> MLXArray {
        guard !samples.isEmpty, channelCount > 0 else {
            throw AudioReaderError.decodeFailed("empty input PCM")
        }
        let stereoPCM = try resampleToStereo44k(
            interleaved: samples,
            sampleRate: sampleRate,
            channelCount: channelCount
        )
        let frames = stereoPCM.count / 2
        var planar = [Float](repeating: 0, count: 2 * frames)
        for i in 0 ..< frames {
            planar[i] = stereoPCM[2 * i]
            planar[frames + i] = stereoPCM[2 * i + 1]
        }
        return MLXArray(planar, [1, 2, frames])
    }

    /// Decodes an audio file to interleaved float PCM in its native sample rate
    /// and channel layout. The caller is responsible for any resampling and
    /// channel-layout adjustments.
    public static func read(url: URL) throws -> (samples: [Float], sampleRate: Int, channelCount: Int) {
        #if canImport(AVFoundation)
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url)
        } catch {
            throw AudioReaderError.decodeFailed(error.localizedDescription)
        }
        let sourceFormat = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)
        guard frameCount > 0 else { throw AudioReaderError.empty(url) }

        guard let buffer = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: frameCount) else {
            throw AudioReaderError.unsupportedFormat(url)
        }
        do {
            try file.read(into: buffer)
        } catch {
            throw AudioReaderError.decodeFailed(error.localizedDescription)
        }

        let channels = Int(sourceFormat.channelCount)
        let frames = Int(buffer.frameLength)
        guard frames > 0 else { throw AudioReaderError.empty(url) }
        guard let floatData = buffer.floatChannelData else {
            throw AudioReaderError.unsupportedFormat(url)
        }

        var interleaved = [Float](repeating: 0, count: frames * channels)
        for c in 0 ..< channels {
            let channelPtr = floatData[c]
            for f in 0 ..< frames {
                interleaved[f * channels + c] = channelPtr[f]
            }
        }
        return (interleaved, Int(sourceFormat.sampleRate.rounded()), channels)
        #else
        throw AudioReaderError.avFoundationUnavailable
        #endif
    }

    private static func resampleToStereo44k(interleaved: [Float], sampleRate: Int, channelCount: Int) throws -> [Float] {
        #if canImport(AVFoundation)
        let targetRate: Double = Double(StableAudioPipeline.sampleRate)
        let frames = interleaved.count / channelCount

        guard let sourceFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: AVAudioChannelCount(channelCount),
            interleaved: false
        ) else {
            throw AudioReaderError.decodeFailed("could not build source AVAudioFormat")
        }
        guard let sourceBuffer = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: AVAudioFrameCount(frames)) else {
            throw AudioReaderError.decodeFailed("could not allocate source buffer")
        }
        sourceBuffer.frameLength = AVAudioFrameCount(frames)
        if let floatData = sourceBuffer.floatChannelData {
            for c in 0 ..< channelCount {
                let dst = floatData[c]
                for f in 0 ..< frames {
                    dst[f] = interleaved[f * channelCount + c]
                }
            }
        }

        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: targetRate,
            channels: 2,
            interleaved: false
        ) else {
            throw AudioReaderError.decodeFailed("could not build target AVAudioFormat")
        }

        guard let converter = AVAudioConverter(from: sourceFormat, to: targetFormat) else {
            throw AudioReaderError.decodeFailed("AVAudioConverter init failed")
        }

        let ratio = targetRate / Double(sampleRate)
        let estimated = AVAudioFrameCount((Double(frames) * ratio).rounded(.up)) + 64
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: estimated) else {
            throw AudioReaderError.decodeFailed("could not allocate output buffer")
        }

        var done = false
        var convertError: NSError?
        let status = converter.convert(to: outputBuffer, error: &convertError) { _, outStatus in
            if done {
                outStatus.pointee = .endOfStream
                return nil
            }
            done = true
            outStatus.pointee = .haveData
            return sourceBuffer
        }

        if let convertError {
            throw AudioReaderError.decodeFailed(convertError.localizedDescription)
        }
        if status == .error {
            throw AudioReaderError.decodeFailed("AVAudioConverter returned error status")
        }

        let outFrames = Int(outputBuffer.frameLength)
        guard outFrames > 0, let outData = outputBuffer.floatChannelData else {
            throw AudioReaderError.decodeFailed("converter produced no output")
        }

        var interleavedOut = [Float](repeating: 0, count: outFrames * 2)
        let left = outData[0]
        let right = outputBuffer.format.channelCount >= 2 ? outData[1] : outData[0]
        for f in 0 ..< outFrames {
            interleavedOut[2 * f] = left[f]
            interleavedOut[2 * f + 1] = right[f]
        }
        return interleavedOut
        #else
        throw AudioReaderError.avFoundationUnavailable
        #endif
    }
}
