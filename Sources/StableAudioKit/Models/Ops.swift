import Foundation
import MLX

func linear(_ x: MLXArray, weight: MLXArray, bias: MLXArray? = nil) -> MLXArray {
    var y = matmul(x, weight.T)
    if let bias {
        y = y + bias
    }
    return y
}

func silu(_ x: MLXArray) -> MLXArray {
    x * sigmoid(x)
}
