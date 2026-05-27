import Foundation

public enum StableAudioModelKind: String, CaseIterable, Identifiable, Sendable, Codable {
    case smallMusic
    case smallSFX
    case medium

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .smallMusic: return "Small Music"
        case .smallSFX: return "Small SFX"
        case .medium: return "Medium"
        }
    }

    var ditResourceName: String {
        switch self {
        case .smallMusic: return "dit_sm-music_f16"
        case .smallSFX: return "dit_sm-sfx_f16"
        case .medium: return "dit_medium_f16"
        }
    }

    var conditionerResourceName: String {
        switch self {
        case .smallMusic: return "sa3_conditioner_sm-music"
        case .smallSFX: return "sa3_conditioner_sm-sfx"
        case .medium: return "sa3_conditioner_medium"
        }
    }

    var autoencoder: StableAudioAutoencoderKind {
        switch self {
        case .smallMusic, .smallSFX: return .sameS
        case .medium: return .sameL
        }
    }

    public var isAvailableOnThisPlatform: Bool {
        switch self {
        case .smallMusic, .smallSFX:
            return true
        case .medium:
            #if os(macOS)
            return true
            #else
            return false
            #endif
        }
    }
}

enum StableAudioAutoencoderKind: Sendable {
    case sameS
    case sameL
}
