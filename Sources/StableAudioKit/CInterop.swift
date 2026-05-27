import Foundation

// C-ABI surface for the StableAudioKit XCFramework. Callable from C/C++/Swift.
// Headers shipped with the framework: StableAudioKit.h

private let lastErrorKey = "com.stableaudiokit.lastErrorBuffer"

private func recordLastError(_ message: String) {
    var bytes = Array(message.utf8)
    bytes.append(0)
    Thread.current.threadDictionary[lastErrorKey] = NSData(bytes: bytes, length: bytes.count)
}

private func clearLastError() {
    Thread.current.threadDictionary.removeObject(forKey: lastErrorKey)
}

@_cdecl("stable_audio_last_error")
public func stable_audio_last_error() -> UnsafePointer<CChar>? {
    guard let data = Thread.current.threadDictionary[lastErrorKey] as? NSData else {
        return nil
    }
    return data.bytes.assumingMemoryBound(to: CChar.self)
}

@_cdecl("stable_audio_pipeline_create")
public func stable_audio_pipeline_create(
    _ weightsDirectoryPath: UnsafePointer<CChar>?
) -> OpaquePointer? {
    clearLastError()
    guard let weightsDirectoryPath else {
        recordLastError("weights_directory_path is NULL")
        return nil
    }
    let path = String(cString: weightsDirectoryPath)
    let url = URL(fileURLWithPath: path, isDirectory: true)
    do {
        let pipeline = try StableAudioPipeline(weightsDirectory: url)
        let retained = Unmanaged.passRetained(pipeline)
        return OpaquePointer(retained.toOpaque())
    } catch {
        recordLastError("\(error)")
        return nil
    }
}

@_cdecl("stable_audio_pipeline_destroy")
public func stable_audio_pipeline_destroy(_ pipelinePtr: OpaquePointer?) {
    guard let pipelinePtr else { return }
    Unmanaged<StableAudioPipeline>.fromOpaque(UnsafeRawPointer(pipelinePtr)).release()
}

public typealias CProgressCallback = @convention(c) (
    Int32, Int32, UnsafePointer<CChar>?, UnsafeMutableRawPointer?
) -> Void

private final class ResultBox<T>: @unchecked Sendable {
    var value: Result<T, Error>?
}

@_cdecl("stable_audio_generate")
public func stable_audio_generate(
    _ pipelinePtr: OpaquePointer?,
    _ modelRaw: Int32,
    _ promptUTF8: UnsafePointer<CChar>?,
    _ durationSeconds: Float,
    _ steps: Int32,
    _ seed: UInt64,
    _ progress: CProgressCallback?,
    _ userData: UnsafeMutableRawPointer?,
    _ outSamples: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    _ outSampleCount: UnsafeMutablePointer<Int>?,
    _ outChannelCount: UnsafeMutablePointer<Int32>?,
    _ outSampleRate: UnsafeMutablePointer<Int32>?,
    _ outElapsedSeconds: UnsafeMutablePointer<Double>?
) -> Int32 {
    clearLastError()
    guard let pipelinePtr, let promptUTF8 else {
        recordLastError("pipeline or prompt is NULL")
        return -1
    }
    let pipeline = Unmanaged<StableAudioPipeline>
        .fromOpaque(UnsafeRawPointer(pipelinePtr))
        .takeUnretainedValue()
    let kind: StableAudioModelKind = modelRaw == 1 ? .smallSFX : .smallMusic
    let request = StableAudioGenerationRequest(
        model: kind,
        prompt: String(cString: promptUTF8),
        seconds: durationSeconds,
        steps: Int(steps),
        seed: seed
    )

    let semaphore = DispatchSemaphore(value: 0)
    let box = ResultBox<StableAudioGenerationResult>()
    let progressFP = progress
    let progressUserDataAddress = userData.map { UInt(bitPattern: $0) }
    Task.detached {
        do {
            let result = try await pipeline.generate(request) { event in
                guard let progressFP else { return }
                switch event {
                case .stage(let name):
                    name.withCString { ptr in
                        progressFP(-1, -1, ptr, progressUserDataAddress.flatMap {
                            UnsafeMutableRawPointer(bitPattern: $0)
                        })
                    }
                case .samplingStep(let index, let total):
                    progressFP(Int32(index), Int32(total), nil, progressUserDataAddress.flatMap {
                        UnsafeMutableRawPointer(bitPattern: $0)
                    })
                }
            }
            box.value = .success(result)
        } catch {
            box.value = .failure(error)
        }
        semaphore.signal()
    }
    semaphore.wait()

    switch box.value! {
    case .success(let result):
        let count = result.samples.count
        let buffer = UnsafeMutablePointer<Float>.allocate(capacity: max(count, 1))
        result.samples.withUnsafeBufferPointer { src in
            if let base = src.baseAddress {
                buffer.initialize(from: base, count: count)
            }
        }
        outSamples?.pointee = buffer
        outSampleCount?.pointee = count
        outChannelCount?.pointee = Int32(result.channelCount)
        outSampleRate?.pointee = Int32(result.sampleRate)
        outElapsedSeconds?.pointee = result.elapsedSeconds
        return 0
    case .failure(let error):
        recordLastError("\(error)")
        return -3
    }
}

@_cdecl("stable_audio_samples_free")
public func stable_audio_samples_free(_ samples: UnsafeMutablePointer<Float>?) {
    guard let samples else { return }
    samples.deallocate()
}

@_cdecl("stable_audio_write_wav")
public func stable_audio_write_wav(
    _ samples: UnsafePointer<Float>?,
    _ sampleCount: Int,
    _ sampleRate: Int32,
    _ outputPathUTF8: UnsafePointer<CChar>?
) -> Int32 {
    clearLastError()
    guard let samples, let outputPathUTF8, sampleCount > 0, sampleRate > 0 else {
        recordLastError("invalid arguments to stable_audio_write_wav")
        return -1
    }
    let path = String(cString: outputPathUTF8)
    let url = URL(fileURLWithPath: path)
    let array = Array(UnsafeBufferPointer(start: samples, count: sampleCount))
    do {
        try AudioWriter.writeStereoSamples(array, sampleRate: Int(sampleRate), to: url)
        return 0
    } catch {
        recordLastError("\(error)")
        return -2
    }
}
