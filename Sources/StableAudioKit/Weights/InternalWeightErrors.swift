import Foundation

enum WeightTensorLoaderError: LocalizedError {
    case missing(String)

    var errorDescription: String? {
        switch self {
        case .missing(let fileName):
            return "\(fileName) missing"
        }
    }
}
