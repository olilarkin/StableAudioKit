import Foundation
import MLX
import XCTest
@testable import StableAudioKit

final class StableAudioKitTests: XCTestCase {
    func testLatentLengthIsEvenAndNonZero() {
        XCTAssertEqual(StableAudioPipeline.latentLength(for: 0.01), 2)
        XCTAssertTrue(StableAudioPipeline.latentLength(for: 10).isMultiple(of: 2))
    }

    func testMissingWeightsAreReported() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("StableAudioKitTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let weights = try StableAudioWeights(directory: directory)
        let statuses = try weights.validate()
        XCTAssertTrue(statuses.contains { $0.fileName == "t5gemma_f16.safetensors" && !$0.isReady })
        XCTAssertThrowsError(try weights.requireReady()) { error in
            XCTAssertTrue(error is StableAudioWeightError)
        }
    }

    func testMissingEncoderIsReported() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("StableAudioKitTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let weights = try StableAudioWeights(directory: directory)
        XCTAssertThrowsError(try weights.requireEncoderReady(for: .smallMusic)) { error in
            XCTAssertTrue(error is StableAudioWeightError)
        }
    }

    func testStartIndexNoiseLevelOne() {
        // initNoiseLevel = 1.0 must start the loop at index 0 — byte-identical
        // to the text-to-audio path.
        let schedule = StableAudioPipeline.buildSchedule(steps: 8)
        XCTAssertEqual(StableAudioPipeline.startIndex(for: 1.0, schedule: schedule), 0)
    }

    func testStartIndexMonotonicWithNoiseLevel() {
        let schedule = StableAudioPipeline.buildSchedule(steps: 8)
        let high = StableAudioPipeline.startIndex(for: 0.95, schedule: schedule)
        let mid = StableAudioPipeline.startIndex(for: 0.5, schedule: schedule)
        let low = StableAudioPipeline.startIndex(for: 0.1, schedule: schedule)
        // Lower noise level should start *later* in the loop (skip more steps).
        XCTAssertLessThanOrEqual(high, mid)
        XCTAssertLessThanOrEqual(mid, low)
    }

    func testRequestStoresInitAudioFields() {
        let request = StableAudioGenerationRequest(
            model: .smallMusic,
            prompt: "test",
            initAudio: .samples(values: [0.0, 0.0], sampleRate: 44_100, channelCount: 1),
            initNoiseLevel: 0.5
        )
        XCTAssertEqual(request.initNoiseLevel, 0.5)
        XCTAssertNotNil(request.initAudio)
    }

    func testAudioReaderRoundTripsSineWave() throws {
        #if canImport(AVFoundation)
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("StableAudioKitTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        // Synthesize a short 440 Hz mono sine wave at 22.05 kHz and write it
        // to disk via AudioWriter (which writes stereo). To get a mono test
        // case we instead feed raw PCM through `loadStereo44k(samples:)`.
        let sampleRate = 22_050
        let durationSeconds = 0.25
        let frames = Int(Double(sampleRate) * durationSeconds)
        var mono = [Float](repeating: 0, count: frames)
        for i in 0 ..< frames {
            mono[i] = sin(2 * .pi * 440 * Float(i) / Float(sampleRate)) * 0.5
        }
        let array = try AudioReader.loadStereo44k(samples: mono, sampleRate: sampleRate, channelCount: 1)
        let shape = array.shape
        XCTAssertEqual(shape.count, 3)
        XCTAssertEqual(shape[0], 1)
        XCTAssertEqual(shape[1], 2)
        // After 22050 -> 44100 resample we should have roughly 2x frames.
        XCTAssertGreaterThan(shape[2], frames)
        XCTAssertLessThan(abs(shape[2] - 2 * frames), 256)
        #else
        throw XCTSkip("AVFoundation not available on this platform")
        #endif
    }
}
