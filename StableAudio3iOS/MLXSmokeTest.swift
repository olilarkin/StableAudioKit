import Foundation
import MLX

enum MLXSmokeTest {
    struct Report {
        let device: String
        let summary: String
    }

    static func run() -> Result<Report, Error> {
        do {
            return .success(try run(on: .gpu))
        } catch {
            do {
                return .success(try run(on: .cpu))
            } catch {
                return .failure(error)
            }
        }
    }

    private static func run(on device: Device) throws -> Report {
        let startedAt = Date()
        let checksum = Stream.withNewDefaultStream(device: device) {
            let size = 64
            let a = MLXArray(0 ..< (size * size), [size, size]).asType(.float32)
            let b = MLXArray(0 ..< (size * size), [size, size]).asType(.float32)
            let c = matmul(a, b)
            let reduced = c.sum()
            eval(c, reduced)
            return reduced.item(Float.self)
        }
        let elapsed = Date().timeIntervalSince(startedAt)
        let summary = "matmul 64x64, checksum \(Int(checksum)), \(String(format: "%.3f", elapsed))s"
        return Report(device: "\(device)", summary: summary)
    }
}

