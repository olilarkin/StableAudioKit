import Foundation

struct WeightManifest: Decodable {
    let model: String
    let format: String
    let files: [WeightFile]
}

struct WeightFile: Decodable, Identifiable {
    var id: String { fileName }

    let role: String
    let fileName: String
    let minimumBytes: Int64
    let sourceFileName: String
}

struct WeightStatus: Identifiable {
    var id: String { fileName }

    let role: String
    let fileName: String
    let expectedBytes: Int64
    let actualBytes: Int64?

    var isReady: Bool {
        guard let actualBytes else { return false }
        return actualBytes >= expectedBytes
    }

    var sizeSummary: String {
        guard let actualBytes else { return "missing" }
        return "\(Self.format(actualBytes)) / \(Self.format(expectedBytes))"
    }

    private static func format(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }
}

