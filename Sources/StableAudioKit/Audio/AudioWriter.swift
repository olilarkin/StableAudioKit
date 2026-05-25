import Foundation
import MLX

public enum AudioWriter {
    public static func write(_ audio: MLXArray, sampleRate: Int = StableAudioPipeline.sampleRate, to url: URL) throws {
        let shape = audio.shape
        precondition(shape.count == 2 && shape[0] == 2)
        let samples = shape[1]
        let values = audio.asArray(Float.self)

        var data = Data()
        appendString("RIFF", to: &data)
        appendUInt32(UInt32(36 + samples * 2 * 2), to: &data)
        appendString("WAVE", to: &data)
        appendString("fmt ", to: &data)
        appendUInt32(16, to: &data)
        appendUInt16(1, to: &data)
        appendUInt16(2, to: &data)
        appendUInt32(UInt32(sampleRate), to: &data)
        appendUInt32(UInt32(sampleRate * 2 * 2), to: &data)
        appendUInt16(4, to: &data)
        appendUInt16(16, to: &data)
        appendString("data", to: &data)
        appendUInt32(UInt32(samples * 2 * 2), to: &data)

        for index in 0 ..< samples {
            appendPCM(values[index], to: &data)
            appendPCM(values[samples + index], to: &data)
        }

        try data.write(to: url, options: [.atomic])
    }

    public static func write(_ result: StableAudioGenerationResult, to url: URL) throws {
        precondition(result.channelCount == 2)
        try writeStereoSamples(result.samples, sampleRate: result.sampleRate, to: url)
    }

    public static func writeStereoSamples(_ values: [Float], sampleRate: Int, to url: URL) throws {
        precondition(values.count.isMultiple(of: 2))
        let samples = values.count / 2

        var data = Data()
        appendString("RIFF", to: &data)
        appendUInt32(UInt32(36 + samples * 2 * 2), to: &data)
        appendString("WAVE", to: &data)
        appendString("fmt ", to: &data)
        appendUInt32(16, to: &data)
        appendUInt16(1, to: &data)
        appendUInt16(2, to: &data)
        appendUInt32(UInt32(sampleRate), to: &data)
        appendUInt32(UInt32(sampleRate * 2 * 2), to: &data)
        appendUInt16(4, to: &data)
        appendUInt16(16, to: &data)
        appendString("data", to: &data)
        appendUInt32(UInt32(samples * 2 * 2), to: &data)

        for index in 0 ..< samples {
            appendPCM(values[index], to: &data)
            appendPCM(values[samples + index], to: &data)
        }

        try data.write(to: url, options: [.atomic])
    }

    private static func appendPCM(_ value: Float, to data: inout Data) {
        let clipped = max(-1.0, min(1.0, value))
        appendInt16(Int16(clipped * 32767.0), to: &data)
    }

    private static func appendString(_ string: String, to data: inout Data) {
        data.append(string.data(using: .ascii)!)
    }

    private static func appendUInt16(_ value: UInt16, to data: inout Data) {
        var little = value.littleEndian
        withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
    }

    private static func appendInt16(_ value: Int16, to data: inout Data) {
        var little = value.littleEndian
        withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
    }

    private static func appendUInt32(_ value: UInt32, to data: inout Data) {
        var little = value.littleEndian
        withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
    }
}
