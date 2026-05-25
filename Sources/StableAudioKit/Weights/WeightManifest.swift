import Foundation

public struct WeightManifest: Decodable, Sendable {
    public let model: String
    public let format: String
    public let files: [WeightFile]
}

public struct WeightFile: Decodable, Identifiable, Sendable {
    public var id: String { fileName }

    public let role: String
    public let fileName: String
    public let minimumBytes: Int64
    public let sourceFileName: String

    public init(role: String, fileName: String, minimumBytes: Int64, sourceFileName: String) {
        self.role = role
        self.fileName = fileName
        self.minimumBytes = minimumBytes
        self.sourceFileName = sourceFileName
    }
}

public struct WeightStatus: Identifiable, Sendable {
    public var id: String { fileName }

    public let role: String
    public let fileName: String
    public let expectedBytes: Int64
    public let actualBytes: Int64?

    public init(role: String, fileName: String, expectedBytes: Int64, actualBytes: Int64?) {
        self.role = role
        self.fileName = fileName
        self.expectedBytes = expectedBytes
        self.actualBytes = actualBytes
    }

    public var isReady: Bool {
        guard let actualBytes else { return false }
        return actualBytes >= expectedBytes
    }

    public var sizeSummary: String {
        guard let actualBytes else { return "missing" }
        return "\(Self.format(actualBytes)) / \(Self.format(expectedBytes))"
    }

    private static func format(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }
}
