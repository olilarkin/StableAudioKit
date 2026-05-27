import Foundation
import MLX

struct SAMELDecoder {
    static let latentDimension = 256
    static let dimension = 1536
    static let headCount = 24
    static let headDimension = 64
    static let ropeDimensions = 32
    static let blockCount = 12
    static let feedForwardInner = 4608
    static let sinStartBlock = 5
    static let outputChannels = 512
    static let stride = 16
    static let subChunkSize = 17           // = stride + 1
    static let blockSize = 17              // SWA query group
    static let swaWindow = 51              // = 3 * blockSize
    static let sinPerPos = 16              // = subChunkSize - 1

    let weights: [String: MLXArray]

    func decodeChunked(latents: MLXArray, chunkSize: Int = 8, overlap: Int = 2) -> MLXArray {
        let latentLength = latents.dim(-1)
        let kernel = chunkSize + 2 * overlap
        if latentLength <= kernel {
            return callAsFunction(latents)
        }

        var pieces: [MLXArray] = []
        let firstOutput = callAsFunction(latents[0..., 0..., 0 ..< kernel])
        let validFirst = chunkSize + overlap
        pieces.append(firstOutput[0..., 0..., 0 ..< (validFirst * Self.stride)])
        var index = validFirst

        while index + chunkSize + overlap <= latentLength {
            let out = callAsFunction(latents[0..., 0..., (index - overlap) ..< (index + chunkSize + overlap)])
            pieces.append(out[0..., 0..., (overlap * Self.stride) ..< ((overlap + chunkSize) * Self.stride)])
            index += chunkSize
        }

        let remaining = latentLength - index
        if remaining > 0 {
            let out = callAsFunction(latents[0..., 0..., (latentLength - kernel) ..< latentLength])
            let start = out.dim(-1) - remaining * Self.stride
            pieces.append(out[0..., 0..., start...])
        }

        return concatenated(pieces, axis: -1)
    }

    func callAsFunction(_ latents: MLXArray) -> MLXArray {
        let batch = latents.dim(0)
        let latentLength = latents.dim(2)

        var x = latents.asType(.float32) * weights["running_std"]!
        x = linear(x.transposed(0, 2, 1), weight: weights["project_in.weight"]!, bias: weights["project_in.bias"]!)

        let expanded = x.expandedDimensions(axis: 2)
        let newTokens = broadcast(weights["new_tokens"]!.expandedDimensions(axis: 0), to: [batch, latentLength, Self.sinPerPos, Self.dimension])
        x = concatenated([expanded, newTokens], axis: 2)
        x = x.reshaped(batch, latentLength * Self.subChunkSize, Self.dimension)

        let totalLength = latentLength * Self.subChunkSize
        let useFullAttention = totalLength <= Self.blockSize
        let swaMask = useFullAttention ? nil : Self.swaMask(dtype: x.dtype)

        for index in 0 ..< Self.blockCount {
            x = block(index: index, x: x, swaMask: swaMask, fullAttention: useFullAttention)
        }

        // Drop original latent slot at index 0 of each 17-group; keep 16
        x = x.reshaped(batch, latentLength, Self.subChunkSize, Self.dimension)
        x = x[0..., 0..., 1..., 0...]
        x = x.reshaped(batch, latentLength * Self.sinPerPos, Self.dimension)

        // mapping: Linear(dim, outputChannels) - apply per token then transpose to channels-first
        let mapped = linear(x, weight: weights["mapping.weight"]!, bias: weights["mapping.bias"]!)
        return mapped.transposed(0, 2, 1)
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
        // Inputs: [B, H, T, D] where T is divisible by blockSize (17).
        // Output: [B, H, T, D]
        let batch = q1.dim(0)
        let heads = q1.dim(1)
        let length = q1.dim(2)
        let dim = q1.dim(3)
        let groups = length / Self.blockSize
        let block = Self.blockSize

        // Group Q into [B, H, G, block, D]
        let q1g = q1.reshaped(batch, heads, groups, block, dim)
        let q2g = q2.reshaped(batch, heads, groups, block, dim)

        // Build windowed K/V: for each group g, concatenate (left-pad-or-prev, current, next-or-right-pad) groups.
        let k1grouped = k1.reshaped(batch, heads, groups, block, dim)
        let k2grouped = k2.reshaped(batch, heads, groups, block, dim)
        let vGrouped = v.reshaped(batch, heads, groups, block, dim)

        let zeroPad = MLXArray.zeros([batch, heads, 1, block, dim], dtype: k1.dtype)
        let leftK1 = concatenated([zeroPad, k1grouped[0..., 0..., 0 ..< (groups - 1), 0..., 0...]], axis: 2)
        let leftK2 = concatenated([zeroPad, k2grouped[0..., 0..., 0 ..< (groups - 1), 0..., 0...]], axis: 2)
        let leftV = concatenated([zeroPad, vGrouped[0..., 0..., 0 ..< (groups - 1), 0..., 0...]], axis: 2)

        let rightK1 = concatenated([k1grouped[0..., 0..., 1 ..< groups, 0..., 0...], zeroPad], axis: 2)
        let rightK2 = concatenated([k2grouped[0..., 0..., 1 ..< groups, 0..., 0...], zeroPad], axis: 2)
        let rightV = concatenated([vGrouped[0..., 0..., 1 ..< groups, 0..., 0...], zeroPad], axis: 2)

        // Concatenate along the block axis to get [B, H, G, 3*block, D] = [B, H, G, 51, D].
        let k1Windowed = concatenated([leftK1, k1grouped, rightK1], axis: 3)
        let k2Windowed = concatenated([leftK2, k2grouped, rightK2], axis: 3)
        let vWindowed = concatenated([leftV, vGrouped, rightV], axis: 3)

        // Boundary mask: zero out attention to the padded left slot for group 0 and padded right slot for group G-1.
        let boundary = Self.boundaryMask(groups: groups, dtype: q1.dtype)
        // boundary: [G, swaWindow]; swaMask: [block, swaWindow]
        // combined shape needed: [G, block, swaWindow]; broadcast block + boundary across groups.
        let combinedMask = swaMask.expandedDimensions(axis: 0) + boundary.expandedDimensions(axis: 1)
        // Need [B*G, 1, block, swaWindow] when we collapse B and G; equivalent broadcast for MLX SDPA.
        // We flatten (B, G) into the batch axis to match the windowed layout.
        let q1Flat = q1g.transposed(0, 2, 1, 3, 4).reshaped(batch * groups, heads, block, dim)
        let q2Flat = q2g.transposed(0, 2, 1, 3, 4).reshaped(batch * groups, heads, block, dim)
        let k1Flat = k1Windowed.transposed(0, 2, 1, 3, 4).reshaped(batch * groups, heads, Self.swaWindow, dim)
        let k2Flat = k2Windowed.transposed(0, 2, 1, 3, 4).reshaped(batch * groups, heads, Self.swaWindow, dim)
        let vFlat = vWindowed.transposed(0, 2, 1, 3, 4).reshaped(batch * groups, heads, Self.swaWindow, dim)

        // Broadcast the mask across batch: [G, block, swaWindow] -> [B*G, 1, block, swaWindow]
        let maskTiled = broadcast(combinedMask.expandedDimensions(axis: 0), to: [batch, groups, block, Self.swaWindow])
            .reshaped(batch * groups, 1, block, Self.swaWindow)

        let scale = pow(Float(Self.headDimension), -0.5)
        let main = scaledDotProductAttention(queries: q1Flat, keys: k1Flat, values: vFlat, scale: scale, mask: maskTiled)
        let diff = scaledDotProductAttention(queries: q2Flat, keys: k2Flat, values: vFlat, scale: scale, mask: maskTiled)
        let diffed = main - diff

        // diffed: [B*G, H, block, D] -> [B, H, T, D]
        return diffed.reshaped(batch, groups, heads, block, dim).transposed(0, 2, 1, 3, 4).reshaped(batch, heads, length, dim)
    }

    private func feedForward(prefix: String, x: MLXArray, useSinGate: Bool) -> MLXArray {
        let projected = linear(x, weight: weights["\(prefix).glu_proj.weight"]!, bias: weights["\(prefix).glu_proj.bias"]!)
        let split = projected.split(parts: 2, axis: -1)
        let value = split[0]
        let gate = split[1]
        let activated: MLXArray
        if useSinGate {
            activated = value * sin(gate * Float.pi)
        } else {
            activated = value * silu(gate)
        }
        return linear(activated, weight: weights["\(prefix).proj_out.weight"]!, bias: weights["\(prefix).proj_out.bias"]!)
    }

    private func dyt(prefix: String, x: MLXArray) -> MLXArray {
        weights["\(prefix).gamma"]! * tanh(weights["\(prefix).alpha"]! * x) + weights["\(prefix).beta"]!
    }

    private func toHeads(_ x: MLXArray, batch: Int, length: Int) -> MLXArray {
        x.reshaped(batch, length, Self.headCount, Self.headDimension).transposed(0, 2, 1, 3)
    }

    static func swaMask(dtype: DType) -> MLXArray {
        // 17x51 mask matching the reference: mask[q, k] = 0 if k in [q, q + 2*blockSize], else -1e9
        var values = Array(repeating: Float(-1e9), count: blockSize * swaWindow)
        for q in 0 ..< blockSize {
            let lo = q
            let hi = q + 2 * blockSize
            for k in lo ... hi {
                values[q * swaWindow + k] = 0
            }
        }
        return MLXArray(values, [blockSize, swaWindow]).asType(dtype)
    }

    static func boundaryMask(groups: Int, dtype: DType) -> MLXArray {
        // For each group g and window position w, absolute padded position is g*blockSize + w.
        // Valid (non-padded) range: [blockSize, blockSize + T) where T = groups * blockSize.
        var values = Array(repeating: Float(0), count: groups * swaWindow)
        for g in 0 ..< groups {
            for w in 0 ..< swaWindow {
                let absolute = g * blockSize + w
                let valid = absolute >= blockSize && absolute < (blockSize + groups * blockSize)
                if !valid {
                    values[g * swaWindow + w] = -1e9
                }
            }
        }
        return MLXArray(values, [groups, swaWindow]).asType(dtype)
    }
}
