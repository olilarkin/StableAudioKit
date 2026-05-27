#ifndef STABLE_AUDIO_KIT_H
#define STABLE_AUDIO_KIT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pipeline handle. Create with stable_audio_pipeline_create,
// release with stable_audio_pipeline_destroy.
typedef struct StableAudioPipeline StableAudioPipeline;

typedef enum StableAudioModel {
    STABLE_AUDIO_MODEL_SMALL_MUSIC = 0,
    STABLE_AUDIO_MODEL_SMALL_SFX = 1,
    // macOS-only; on iOS/visionOS, requesting this model causes
    // stable_audio_generate to fail with STABLE_AUDIO_ERR_RT.
    STABLE_AUDIO_MODEL_MEDIUM = 2
} StableAudioModel;

// Status codes returned by stable_audio_generate and stable_audio_write_wav.
// 0 means success; negative values indicate failure (call stable_audio_last_error).
#define STABLE_AUDIO_OK         0
#define STABLE_AUDIO_ERR_ARG   -1
#define STABLE_AUDIO_ERR_IO    -2
#define STABLE_AUDIO_ERR_RT    -3

// Progress callback. Either step_index/step_total are >= 0 and stage_name is NULL
// (sampling progress), or step_index/step_total are -1 and stage_name is non-NULL
// (lifecycle stage event). May be invoked from a background thread.
typedef void (*StableAudioProgressCallback)(
    int32_t step_index,
    int32_t step_total,
    const char *stage_name,
    void *user_data);

// Returns the last error message for the calling thread, or NULL if none.
// The pointer is owned by the library and remains valid until the next
// call into the library on this thread.
const char *stable_audio_last_error(void);

// Load the pipeline using prepared weights at weights_directory_path
// (the directory produced by Scripts/prepare_weights.py). Returns NULL on failure;
// inspect stable_audio_last_error() for details.
StableAudioPipeline *stable_audio_pipeline_create(
    const char *weights_directory_path);

// Releases the pipeline. NULL is a no-op.
void stable_audio_pipeline_destroy(StableAudioPipeline *pipeline);

// Run text-to-audio generation. On success returns 0 and writes a newly
// allocated PCM-float32 buffer to *out_samples (free with stable_audio_samples_free),
// along with the buffer size and audio metadata. The samples are PLANAR stereo:
// the first (*out_sample_count / 2) values are the left channel, the next half
// are the right channel.
int32_t stable_audio_generate(
    StableAudioPipeline *pipeline,
    StableAudioModel model,
    const char *prompt_utf8,
    float duration_seconds,
    int32_t steps,
    uint64_t seed,
    StableAudioProgressCallback progress,
    void *user_data,
    float **out_samples,
    size_t *out_sample_count,
    int32_t *out_channel_count,
    int32_t *out_sample_rate,
    double *out_elapsed_seconds);

// Free a buffer returned by stable_audio_generate. NULL is a no-op.
void stable_audio_samples_free(float *samples);

// Write a stereo float32 buffer (planar layout, same convention as
// stable_audio_generate) to a 16-bit PCM WAV file at output_path_utf8.
int32_t stable_audio_write_wav(
    const float *samples,
    size_t sample_count,
    int32_t sample_rate,
    const char *output_path_utf8);

#ifdef __cplusplus
}
#endif

#endif // STABLE_AUDIO_KIT_H
