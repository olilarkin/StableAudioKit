import Foundation
import MLX

struct StableAudioGenerator {
    enum Stage: String {
        case textEncoder = "T5Gemma"
        case diffusionTransformer = "DiT"
        case decoder = "same-s"
        case sampler = "Euler"
    }

    let stages: [Stage] = [
        .textEncoder,
        .diffusionTransformer,
        .decoder,
        .sampler,
    ]
}

