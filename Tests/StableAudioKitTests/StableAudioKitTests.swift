import Foundation
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
}
