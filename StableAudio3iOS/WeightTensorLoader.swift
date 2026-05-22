import Foundation
import MLX

enum WeightTensorLoader {
    struct Report {
        let tensorCount: Int
        let sample: String
    }

    static func load(fileName: String) -> Result<Report, Error> {
        do {
            guard let url = Bundle.main.url(
                forResource: (fileName as NSString).deletingPathExtension,
                withExtension: (fileName as NSString).pathExtension,
                subdirectory: "Weights"
            ) else {
                throw WeightTensorLoaderError.missing(fileName)
            }

            let arrays = try loadArrays(url: url, stream: .cpu)
            let firstKey = arrays.keys.sorted().first ?? "-"
            let firstShape = arrays[firstKey]?.shape.map(String.init).joined(separator: "x") ?? "-"
            return .success(Report(tensorCount: arrays.count, sample: "\(firstKey) [\(firstShape)]"))
        } catch {
            return .failure(error)
        }
    }
}

enum WeightTensorLoaderError: LocalizedError {
    case missing(String)

    var errorDescription: String? {
        switch self {
        case .missing(let fileName):
            return "\(fileName) missing"
        }
    }
}

