import Foundation
import MLX

// SAME-S encoder: the structural inverse of `SAMESDecoder`. Both share the same
// transformer block layout (6 layers, 768 dim, 12 heads, RoPE 32, differential
// attention, DYT pre/post-norm, GLU+SiLU FFN) and the same effective-chunk
// shift scheme used by the decoder. The differences mirror the upstream
// `TransformerResamplingBlock(type='encoder')` forward:
//
//  1. Input `mapping` Conv1d is applied at the *start* (in_channels=2 audio
//     -> dim), where the decoder applies it at the *end*.
//  2. `new_tokens` are appended (one per stride-group, shape [1, 1, dim])
//     and only the *new_token* slot is kept after the transformer stack
//     (the decoder keeps the opposite slots).
//  3. An output `project_out` Linear maps dim -> latent_dim (256).
//  4. The final tensor is divided by `running_std` to match the bottleneck
//     scale used at decode time.
struct SAMESEncoder {
    static let latentDimension = SAMESDecoder.latentDimension
    static let dimension = SAMESDecoder.dimension
    static let headCount = SAMESDecoder.headCount
    static let headDimension = SAMESDecoder.headDimension
    static let ropeDimensions = SAMESDecoder.ropeDimensions
    static let blockCount = SAMESDecoder.blockCount
    static let feedForwardInner = SAMESDecoder.feedForwardInner
    static let inputChannels = SAMESDecoder.outputChannels  // 512 (= 2 stereo * samplesPerStrideFrame)
    static let stride = SAMESDecoder.stride
    static let subChunkSize = SAMESDecoder.subChunkSize
    static let chunkSizeLatent = SAMESDecoder.chunkSizeLatent
    static let effectiveChunk = SAMESDecoder.effectiveChunk
    static let shift = SAMESDecoder.shift
    static let samplesPerStrideFrame = inputChannels / 2  // 256
    static let samplesPerLatent = stride * samplesPerStrideFrame  // 4096

    let weights: [String: MLXArray]

    func encodeChunked(audio: MLXArray, chunkSize: Int = 8, overlap: Int = 2) -> MLXArray {
        let audioLength = audio.dim(-1)
        let perLatent = Self.samplesPerLatent
        precondition(audioLength % perLatent == 0, "audio length must be a multiple of samplesPerLatent")
        let latentLength = audioLength / perLatent
        let kernel = chunkSize + 2 * overlap
        if latentLength <= kernel {
            return callAsFunction(audio)
        }

        var pieces: [MLXArray] = []
        let firstOutput = callAsFunction(audio[0..., 0..., 0 ..< (kernel * perLatent)])
        let validFirst = chunkSize + overlap
        pieces.append(firstOutput[0..., 0..., 0 ..< validFirst])
        var index = validFirst

        while index + chunkSize + overlap <= latentLength {
            let start = (index - overlap) * perLatent
            let stop = (index + chunkSize + overlap) * perLatent
            let out = callAsFunction(audio[0..., 0..., start ..< stop])
            pieces.append(out[0..., 0..., overlap ..< (overlap + chunkSize)])
            index += chunkSize
        }

        let remaining = latentLength - index
        if remaining > 0 {
            let start = (latentLength - kernel) * perLatent
            let stop = latentLength * perLatent
            let out = callAsFunction(audio[0..., 0..., start ..< stop])
            let outStart = out.dim(-1) - remaining
            pieces.append(out[0..., 0..., outStart...])
        }

        return concatenated(pieces, axis: -1)
    }

    func callAsFunction(_ audio: MLXArray) -> MLXArray {
        let batch = audio.dim(0)
        let audioLength = audio.dim(2)
        let latentLength = audioLength / Self.samplesPerLatent

        // Inverse of `patchedDecode`: [B, 2, L*stride*spf] -> [B, 2*spf, L*stride]
        var x = audio.asType(.float32)
        x = x.reshaped(batch, 2, latentLength * Self.stride, Self.samplesPerStrideFrame)
        x = x.transposed(0, 1, 3, 2)
        x = x.reshaped(batch, Self.inputChannels, latentLength * Self.stride)

        // Input mapping is stored as a kernel=1 Conv1d flattened to Linear by
        // the upstream export (shape [dim, inputChannels]).
        x = x.transposed(0, 2, 1)  // [B, L*stride, inputChannels]
        x = linear(x, weight: weights["mapping.weight"]!, bias: weights["mapping.bias"]!)
        // x: [B, L*stride, dim]

        // Append the encoder's single new_token slot to each stride-group.
        let internalLength = latentLength * Self.subChunkSize
        x = x.reshaped(batch, latentLength, Self.stride, Self.dimension)
        let newTokens = broadcast(
            weights["new_tokens"]!.expandedDimensions(axis: 0),
            to: [batch, latentLength, 1, Self.dimension]
        )
        x = concatenated([x, newTokens], axis: 2)
        x = x.reshaped(batch, internalLength, Self.dimension)

        let firstChunks = internalLength / Self.effectiveChunk
        x = x.reshaped(batch * firstChunks, Self.effectiveChunk, Self.dimension)
        for index in 0 ..< 3 {
            x = block(index: index, x: x)
        }
        x = x.reshaped(batch, internalLength, Self.dimension)

        let left = x[0..., 0 ..< Self.shift, 0...]
        let right = x[0..., (internalLength - Self.shift) ..< internalLength, 0...]
        x = concatenated([left, x, right], axis: 1)
        let secondChunks = (internalLength + Self.effectiveChunk) / Self.effectiveChunk
        x = x.reshaped(batch * secondChunks, Self.effectiveChunk, Self.dimension)
        for index in 3 ..< Self.blockCount {
            x = block(index: index, x: x)
        }
        x = x.reshaped(batch, internalLength + Self.effectiveChunk, Self.dimension)
        x = x[0..., Self.shift ..< (internalLength + Self.shift), 0...]

        // Keep only the new_token output slot of each (stride+1) group.
        x = x.reshaped(batch * latentLength, Self.subChunkSize, Self.dimension)
        x = x[0..., (Self.subChunkSize - 1)..., 0...]
        x = x.reshaped(batch, latentLength, Self.dimension)

        // Output projection: dim -> latent_dim
        var out = linear(x, weight: weights["project_out.weight"]!, bias: weights["project_out.bias"]!)
        out = out.transposed(0, 2, 1)

        // Bottleneck normalization (inverse of decoder's `latents * running_std`).
        out = out / weights["running_std"]!
        return out
    }

    private func block(index: Int, x: MLXArray) -> MLXArray {
        let prefix = "blocks.\(index)"
        var out = x + attention(prefix: "\(prefix).attn", x: dyt(prefix: "\(prefix).pre_norm", x: x))
        out = out + feedForward(prefix: "\(prefix).ff", x: dyt(prefix: "\(prefix).ff_norm", x: out))
        return out
    }

    private func attention(prefix: String, x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let qkv = linear(x, weight: weights["\(prefix).to_qkv.weight"]!)
        let split = qkv.split(parts: 5, axis: -1)
        var q = toHeads(split[0], batch: batch, length: length)
        var k = toHeads(split[1], batch: batch, length: length)
        let v = toHeads(split[2], batch: batch, length: length)
        var qDiff = toHeads(split[3], batch: batch, length: length)
        var kDiff = toHeads(split[4], batch: batch, length: length)

        q = dyt(prefix: "\(prefix).q_norm", x: q)
        k = dyt(prefix: "\(prefix).k_norm", x: k)
        qDiff = dyt(prefix: "\(prefix).q_norm", x: qDiff)
        kDiff = dyt(prefix: "\(prefix).k_norm", x: kDiff)

        q = RoPE(q, dimensions: Self.ropeDimensions, traditional: false, base: 10_000, scale: 1, offset: 0)
        k = RoPE(k, dimensions: Self.ropeDimensions, traditional: false, base: 10_000, scale: 1, offset: 0)
        qDiff = RoPE(qDiff, dimensions: Self.ropeDimensions, traditional: false, base: 10_000, scale: 1, offset: 0)
        kDiff = RoPE(kDiff, dimensions: Self.ropeDimensions, traditional: false, base: 10_000, scale: 1, offset: 0)

        let scale = pow(Float(Self.headDimension), -0.5)
        let main = scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
        let diff = scaledDotProductAttention(queries: qDiff, keys: kDiff, values: v, scale: scale, mask: nil)
        let out = (main - diff).transposed(0, 2, 1, 3).reshaped(batch, length, Self.dimension)
        return linear(out, weight: weights["\(prefix).to_out.weight"]!)
    }

    private func feedForward(prefix: String, x: MLXArray) -> MLXArray {
        let projected = linear(x, weight: weights["\(prefix).glu_proj.weight"]!, bias: weights["\(prefix).glu_proj.bias"]!)
        let split = projected.split(parts: 2, axis: -1)
        let activated = split[0] * silu(split[1])
        return linear(activated, weight: weights["\(prefix).proj_out.weight"]!, bias: weights["\(prefix).proj_out.bias"]!)
    }

    private func dyt(prefix: String, x: MLXArray) -> MLXArray {
        weights["\(prefix).gamma"]! * tanh(weights["\(prefix).alpha"]! * x) + weights["\(prefix).beta"]!
    }

    private func toHeads(_ x: MLXArray, batch: Int, length: Int) -> MLXArray {
        x.reshaped(batch, length, Self.headCount, Self.headDimension).transposed(0, 2, 1, 3)
    }
}
