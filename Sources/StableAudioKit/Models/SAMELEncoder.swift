import Foundation
import MLX

// SAME-L encoder: structural inverse of `SAMELDecoder` with the same 12-layer
// differential sliding-window attention stack. See the docs at the top of
// `SAMESEncoder.swift` for the high-level mirroring contract (input mapping
// first, append one new_token per stride-group, keep the new_token slot, then
// project dim -> latent_dim and divide by running_std).
struct SAMELEncoder {
    static let latentDimension = SAMELDecoder.latentDimension
    static let dimension = SAMELDecoder.dimension
    static let headCount = SAMELDecoder.headCount
    static let headDimension = SAMELDecoder.headDimension
    static let ropeDimensions = SAMELDecoder.ropeDimensions
    static let blockCount = SAMELDecoder.blockCount
    static let feedForwardInner = SAMELDecoder.feedForwardInner
    static let sinStartBlock = SAMELDecoder.sinStartBlock
    static let inputChannels = SAMELDecoder.outputChannels
    static let stride = SAMELDecoder.stride
    static let subChunkSize = SAMELDecoder.subChunkSize
    static let blockSize = SAMELDecoder.blockSize
    static let swaWindow = SAMELDecoder.swaWindow
    static let sinPerPos = SAMELDecoder.sinPerPos
    static let samplesPerStrideFrame = inputChannels / 2
    static let samplesPerLatent = stride * samplesPerStrideFrame

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

        // Inverse of patchedDecode.
        var x = audio.asType(.float32)
        x = x.reshaped(batch, 2, latentLength * Self.stride, Self.samplesPerStrideFrame)
        x = x.transposed(0, 1, 3, 2)
        x = x.reshaped(batch, Self.inputChannels, latentLength * Self.stride)

        // Input mapping is stored as a kernel=1 Conv1d flattened to Linear by
        // prepare_weights.py (see comment in `convert_file`), so a per-token
        // matmul matches the decoder's symmetric end-mapping.
        x = x.transposed(0, 2, 1)  // [B, L*stride, inputChannels]
        x = linear(x, weight: weights["mapping.weight"]!, bias: weights["mapping.bias"]!)
        // x: [B, L*stride, dim]

        let internalLength = latentLength * Self.subChunkSize
        x = x.reshaped(batch, latentLength, Self.stride, Self.dimension)
        let newTokens = broadcast(
            weights["new_tokens"]!.expandedDimensions(axis: 0),
            to: [batch, latentLength, 1, Self.dimension]
        )
        x = concatenated([x, newTokens], axis: 2)
        x = x.reshaped(batch, internalLength, Self.dimension)

        let totalLength = internalLength
        let useFullAttention = totalLength <= Self.blockSize
        let swaMask = useFullAttention ? nil : Self.swaMask(dtype: x.dtype)

        for index in 0 ..< Self.blockCount {
            x = block(index: index, x: x, swaMask: swaMask, fullAttention: useFullAttention)
        }

        // Keep only the new_token slot of each subChunkSize-group.
        x = x.reshaped(batch, latentLength, Self.subChunkSize, Self.dimension)
        x = x[0..., 0..., (Self.subChunkSize - 1)..., 0...]
        x = x.reshaped(batch, latentLength, Self.dimension)

        // Output projection dim -> latent_dim.
        var out = linear(x, weight: weights["project_out.weight"]!, bias: weights["project_out.bias"]!)
        out = out.transposed(0, 2, 1)

        out = out / weights["running_std"]!
        return out
    }

    private func block(index: Int, x: MLXArray, swaMask: MLXArray?, fullAttention: Bool) -> MLXArray {
        let prefix = "blocks.\(index)"
        let useSinGate = index >= Self.sinStartBlock
        var out = x + differentialSWA(prefix: "\(prefix).attn", x: dyt(prefix: "\(prefix).pre_norm", x: x), swaMask: swaMask, fullAttention: fullAttention)
        out = out + feedForward(prefix: "\(prefix).ff", x: dyt(prefix: "\(prefix).ff_norm", x: out), useSinGate: useSinGate)
        return out
    }

    private func differentialSWA(prefix: String, x: MLXArray, swaMask: MLXArray?, fullAttention: Bool) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let qkv = linear(x, weight: weights["\(prefix).to_qkv.weight"]!)
        let split = qkv.split(parts: 5, axis: -1)
        var q1 = toHeads(split[0], batch: batch, length: length)
        var k1 = toHeads(split[1], batch: batch, length: length)
        let v = toHeads(split[2], batch: batch, length: length)
        var q2 = toHeads(split[3], batch: batch, length: length)
        var k2 = toHeads(split[4], batch: batch, length: length)

        q1 = dyt(prefix: "\(prefix).q_norm", x: q1)
        k1 = dyt(prefix: "\(prefix).k_norm", x: k1)
        q2 = dyt(prefix: "\(prefix).q_norm", x: q2)
        k2 = dyt(prefix: "\(prefix).k_norm", x: k2)

        q1 = RoPE(q1, dimensions: Self.ropeDimensions, traditional: false, base: 10_000, scale: 1, offset: 0)
        k1 = RoPE(k1, dimensions: Self.ropeDimensions, traditional: false, base: 10_000, scale: 1, offset: 0)
        q2 = RoPE(q2, dimensions: Self.ropeDimensions, traditional: false, base: 10_000, scale: 1, offset: 0)
        k2 = RoPE(k2, dimensions: Self.ropeDimensions, traditional: false, base: 10_000, scale: 1, offset: 0)

        let out: MLXArray
        if fullAttention || swaMask == nil {
            out = diffFullSDPA(q1: q1, k1: k1, v: v, q2: q2, k2: k2)
        } else {
            out = diffSWA(q1: q1, k1: k1, v: v, q2: q2, k2: k2, swaMask: swaMask!)
        }

        return linear(out.transposed(0, 2, 1, 3).reshaped(batch, length, Self.dimension), weight: weights["\(prefix).to_out.weight"]!)
    }

    private func diffFullSDPA(q1: MLXArray, k1: MLXArray, v: MLXArray, q2: MLXArray, k2: MLXArray) -> MLXArray {
        let scale = pow(Float(Self.headDimension), -0.5)
        let main = scaledDotProductAttention(queries: q1, keys: k1, values: v, scale: scale, mask: nil)
        let diff = scaledDotProductAttention(queries: q2, keys: k2, values: v, scale: scale, mask: nil)
        return main - diff
    }

    private func diffSWA(q1: MLXArray, k1: MLXArray, v: MLXArray, q2: MLXArray, k2: MLXArray, swaMask: MLXArray) -> MLXArray {
        let batch = q1.dim(0)
        let heads = q1.dim(1)
        let length = q1.dim(2)
        let dim = q1.dim(3)
        let groups = length / Self.blockSize
        let blk = Self.blockSize

        let q1g = q1.reshaped(batch, heads, groups, blk, dim)
        let q2g = q2.reshaped(batch, heads, groups, blk, dim)

        let k1grouped = k1.reshaped(batch, heads, groups, blk, dim)
        let k2grouped = k2.reshaped(batch, heads, groups, blk, dim)
        let vGrouped = v.reshaped(batch, heads, groups, blk, dim)

        let zeroPad = MLXArray.zeros([batch, heads, 1, blk, dim], dtype: k1.dtype)
        let leftK1 = concatenated([zeroPad, k1grouped[0..., 0..., 0 ..< (groups - 1), 0..., 0...]], axis: 2)
        let leftK2 = concatenated([zeroPad, k2grouped[0..., 0..., 0 ..< (groups - 1), 0..., 0...]], axis: 2)
        let leftV = concatenated([zeroPad, vGrouped[0..., 0..., 0 ..< (groups - 1), 0..., 0...]], axis: 2)

        let rightK1 = concatenated([k1grouped[0..., 0..., 1 ..< groups, 0..., 0...], zeroPad], axis: 2)
        let rightK2 = concatenated([k2grouped[0..., 0..., 1 ..< groups, 0..., 0...], zeroPad], axis: 2)
        let rightV = concatenated([vGrouped[0..., 0..., 1 ..< groups, 0..., 0...], zeroPad], axis: 2)

        let k1Windowed = concatenated([leftK1, k1grouped, rightK1], axis: 3)
        let k2Windowed = concatenated([leftK2, k2grouped, rightK2], axis: 3)
        let vWindowed = concatenated([leftV, vGrouped, rightV], axis: 3)

        let boundary = SAMELDecoder.boundaryMask(groups: groups, dtype: q1.dtype)
        let combinedMask = swaMask.expandedDimensions(axis: 0) + boundary.expandedDimensions(axis: 1)
        let q1Flat = q1g.transposed(0, 2, 1, 3, 4).reshaped(batch * groups, heads, blk, dim)
        let q2Flat = q2g.transposed(0, 2, 1, 3, 4).reshaped(batch * groups, heads, blk, dim)
        let k1Flat = k1Windowed.transposed(0, 2, 1, 3, 4).reshaped(batch * groups, heads, Self.swaWindow, dim)
        let k2Flat = k2Windowed.transposed(0, 2, 1, 3, 4).reshaped(batch * groups, heads, Self.swaWindow, dim)
        let vFlat = vWindowed.transposed(0, 2, 1, 3, 4).reshaped(batch * groups, heads, Self.swaWindow, dim)

        let maskTiled = broadcast(combinedMask.expandedDimensions(axis: 0), to: [batch, groups, blk, Self.swaWindow])
            .reshaped(batch * groups, 1, blk, Self.swaWindow)

        let scale = pow(Float(Self.headDimension), -0.5)
        let main = scaledDotProductAttention(queries: q1Flat, keys: k1Flat, values: vFlat, scale: scale, mask: maskTiled)
        let diff = scaledDotProductAttention(queries: q2Flat, keys: k2Flat, values: vFlat, scale: scale, mask: maskTiled)
        let diffed = main - diff

        return diffed.reshaped(batch, groups, heads, blk, dim).transposed(0, 2, 1, 3, 4).reshaped(batch, heads, length, dim)
    }

    private func feedForward(prefix: String, x: MLXArray, useSinGate: Bool) -> MLXArray {
        let projected = linear(x, weight: weights["\(prefix).glu_proj.weight"]!, bias: weights["\(prefix).glu_proj.bias"]!)
        let split = projected.split(parts: 2, axis: -1)
        let value = split[0]
        let gate = split[1]
        let activated = useSinGate ? value * sin(gate * Float.pi) : value * silu(gate)
        return linear(activated, weight: weights["\(prefix).proj_out.weight"]!, bias: weights["\(prefix).proj_out.bias"]!)
    }

    private func dyt(prefix: String, x: MLXArray) -> MLXArray {
        weights["\(prefix).gamma"]! * tanh(weights["\(prefix).alpha"]! * x) + weights["\(prefix).beta"]!
    }

    private func toHeads(_ x: MLXArray, batch: Int, length: Int) -> MLXArray {
        x.reshaped(batch, length, Self.headCount, Self.headDimension).transposed(0, 2, 1, 3)
    }

    static func swaMask(dtype: DType) -> MLXArray {
        SAMELDecoder.swaMask(dtype: dtype)
    }
}
