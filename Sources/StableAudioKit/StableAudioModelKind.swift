import Foundation

public enum StableAudioModelKind: String, CaseIterable, Identifiable, Sendable, Codable {
    case smallMusic
    case smallSFX

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .smallMusic: return "Small Music"
        case .smallSFX: return "Small SFX"
        }
    }

    var ditResourceName: String {
        switch self {
        case .smallMusic: return "dit_sm-music_f16"
        case .smallSFX: return "dit_sm-sfx_f16"
        }
    }

    var conditionerResourceName: String {
        switch self {
        case .smallMusic: return "sa3_conditioner_sm-music"
        case .smallSFX: return "sa3_conditioner_sm-sfx"
        }
    }
}
