import Foundation
import MLX

/// Points MLX at the `mlx.metallib` GPU shader library when it is shipped inside
/// this distribution's framework bundle.
///
/// In the SwiftPM / CLI build the metallib sits next to the executable and MLX's
/// default search locates it, so this is a no-op. In the binary `.xcframework`,
/// MLX is linked externally from the consumer's own mlx-swift and has no compiled
/// metallib of its own (SwiftPM does not compile `.metal`); the shaders are
/// embedded in the framework bundle instead, so we hand MLX an explicit path
/// before any GPU work runs.
///
/// Because the externalized xcframework ships a *static* framework, its code is
/// linked into the host binary and `Bundle(for:)` resolves to the host, not to
/// `StableAudioKit.framework`. We therefore scan the likely bundles for the
/// metallib rather than assuming a single location.
public enum MLXRuntime {
    private final class BundleAnchor {}

    private static let bootstrap: Void = {
        if let path = locateMetallib() {
            GPU.setMetallibPath(path)
        }
    }()

    /// Runs the one-time metallib lookup. Idempotent and cheap to call repeatedly.
    static func ensureConfigured() { _ = bootstrap }

    /// Explicitly point MLX at an `mlx.metallib`. Use this from hosts where the
    /// framework bundle is not on a standard search path (e.g. an audio plugin
    /// loaded by a DAW). Must be called before the first `StableAudioPipeline`
    /// operation.
    public static func setMetallibPath(_ path: String) {
        GPU.setMetallibPath(path)
    }

    private static func locateMetallib() -> String? {
        var candidates: [Bundle] = [Bundle(for: BundleAnchor.self), .main]
        candidates.append(contentsOf: Bundle.allFrameworks)
        for bundle in candidates {
            if let url = bundle.url(forResource: "mlx", withExtension: "metallib") {
                return url.path
            }
        }
        // Fall back to an embedded StableAudioKit.framework next to the host.
        let fm = FileManager.default
        for root in [Bundle(for: BundleAnchor.self).bundleURL, Bundle.main.bundleURL] {
            let path = root
                .appendingPathComponent("Frameworks/StableAudioKit.framework/mlx.metallib")
            if fm.fileExists(atPath: path.path) {
                return path.path
            }
        }
        return nil
    }
}
