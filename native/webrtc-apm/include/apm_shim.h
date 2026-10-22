/*
 * webrtc-apm — C ABI shim around the WebRTC M151 Audio Processing Module (APM).
 *
 * Wraps a SINGLE webrtc::AudioProcessing instance configured as a unified
 * capture-side processor: High-Pass Filter -> AEC3 -> Noise Suppression ->
 * AGC2, driven from the host's real-time audio callback. The reverse (render /
 * far-end) stream is the speaker playout used as the AEC reference.
 *
 * Design rules (do not break — see plan):
 *   - Pure C ABI. No C++/STL types cross this boundary. This lets the managed
 *     side P/Invoke it identically on every platform, and isolates the caller
 *     from WebRTC's bundled libc++ ABI (std::__Cr:: inline namespace on
 *     Linux/macOS builds).
 *   - Every entry point is noexcept and returns an int status code. C++
 *     exceptions are swallowed at the boundary and turned into APM_ERR_INTERNAL.
 *     A native crash on the real-time thread would tear down the host process,
 *     so the shim must never let one escape.
 *   - Audio buffers are deinterleaved float, range [-1, 1], exactly
 *     `num_frames` samples per channel where num_frames == sample_rate / 100
 *     (one ~10 ms APM chunk). Re-blocking to this size is the caller's job.
 *
 * Status codes: 0 == success. Positive values are unused. Negative values in
 * the range [-99, -1] are passed through verbatim from WebRTC
 * (webrtc::AudioProcessing::Error, e.g. -8 == kBadDataLengthError). Negative
 * values <= -100 originate in this shim (see APM_ERR_* below).
 */
#ifndef VOICEBOX_WEBRTC_APM_SHIM_H_
#define VOICEBOX_WEBRTC_APM_SHIM_H_

#include <stdint.h>

#if defined(_WIN32)
#define APM_API __declspec(dllexport)
#define APM_CALL __cdecl
#else
#define APM_API __attribute__((visibility("default")))
#define APM_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ABI version of this header/shim contract. The managed side reads
 * apm_abi_version() at load time and refuses to run against a mismatched
 * native binary, so a stale committed library fails loudly instead of
 * crashing. Bump whenever ApmConfig/ApmStats layout or function signatures
 * change in an incompatible way.
 *
 * v2: ApmStats gained residual_echo_likelihood_recent_max and
 *     divergent_filter_fraction (both appended at the end).
 * v3: added the standalone AEC3 delay-estimator API (apm_delay_*), an opaque
 *     handle independent of the APM handle. Existing ApmConfig/ApmStats layout
 *     and apm_* signatures are unchanged.
 * v4: added apm_create_with_neural_residual_echo_model(), which injects the
 *     WebRTC ML residual-echo estimator from a caller-provided TFLite model. */
#define APM_ABI_VERSION 4

/* Shim-originated status codes (WebRTC's own errors pass through in [-99,-1]). */
#define APM_OK                0
#define APM_ERR_NULL_ARG   (-100) /* a required pointer argument was null      */
#define APM_ERR_ABI        (-101) /* ApmConfig.struct_size / version mismatch  */
#define APM_ERR_INTERNAL   (-102) /* a C++ exception was caught at the boundary */
#define APM_ERR_CREATE     (-103) /* AudioProcessing builder returned null      */
#define APM_ERR_BAD_ARG    (-104) /* an argument was out of the supported range */
#define APM_ERR_MODEL      (-105) /* TFLite model or neural estimator invalid   */

/* Noise-suppression aggressiveness. Mirrors
 * webrtc::AudioProcessing::Config::NoiseSuppression::Level. */
typedef enum ApmNsLevel {
  APM_NS_LOW = 0,
  APM_NS_MODERATE = 1,
  APM_NS_HIGH = 2,
  APM_NS_VERY_HIGH = 3
} ApmNsLevel;

/*
 * Creation-time configuration. The caller must set `struct_size` to
 * sizeof(ApmConfig) so the shim can detect a layout mismatch and, in future
 * versions, distinguish old/new callers. All other fields map to APM config.
 *
 * The double-talk tuning fields (dt_* and ep_nearend_len) belong to
 * webrtc::EchoCanceller3Config (a SEPARATE struct from AudioProcessing::Config)
 * and are injected via the builder's SetEchoCancellerConfig() before Build().
 * They are the knobs that govern near-end preservation during double-talk.
 * Defaults below match the WebRTC M151 EchoCanceller3Config defaults; pass
 * those values for stock behavior.
 */
typedef struct ApmConfig {
  uint32_t struct_size;        /* MUST be set to sizeof(ApmConfig).            */

  int32_t sample_rate;         /* 8000 / 16000 / 32000 / 48000.               */
  int32_t num_channels;        /* M1: must be 1 (mono).                       */

  /* Top-level APM submodule enables (webrtc::AudioProcessing::Config). */
  int32_t enable_high_pass_filter;   /* bool: HPF before AEC.                 */
  int32_t enable_echo_canceller;     /* bool: AEC3 (desktop).                 */
  int32_t enable_noise_suppression;  /* bool.                                */
  int32_t ns_level;                  /* ApmNsLevel (used iff NS enabled).    */
  int32_t enable_agc2;               /* bool: modern adaptive gain + limiter.*/
  int32_t agc2_adaptive_digital;     /* bool: AGC2 adaptive-digital stage.   */

  /* Initial render->capture round-trip delay hint, ms. AEC3 still adapts; this
   * is just a starting point (set_stream_delay_ms). */
  int32_t stream_delay_ms;

  /* EchoCanceller3Config double-talk tuning (defaults shown). */
  float dt_enr_threshold;       /* dominant_nearend_detection.enr_threshold      = .25f  */
  float dt_enr_exit_threshold;  /* dominant_nearend_detection.enr_exit_threshold = 10.f  */
  float dt_snr_threshold;       /* dominant_nearend_detection.snr_threshold      = 30.f  */
  float ep_nearend_len;         /* ep_strength.nearend_len                       = 0.83f */
} ApmConfig;

/*
 * Snapshot of APM statistics. Each metric is an (has_x, x) pair because the
 * underlying webrtc::AudioProcessingStats fields are std::optional and may be
 * absent (e.g. before AEC convergence). When has_x == 0, x is undefined.
 */
typedef struct ApmStats {
  int32_t has_echo_return_loss;
  double  echo_return_loss;                 /* ERL  = 10log10(P_far / P_echo)  */

  int32_t has_echo_return_loss_enhancement;
  double  echo_return_loss_enhancement;     /* ERLE = 10log10(P_echo / P_out)  */

  int32_t has_delay_ms;
  int32_t delay_ms;                         /* instantaneous AEC delay estimate */

  int32_t has_delay_median_ms;
  int32_t delay_median_ms;

  int32_t has_residual_echo_likelihood;
  double  residual_echo_likelihood;

  /* ABI v2 additions (appended to preserve the v1 field offsets). Both are
   * public webrtc::AudioProcessingStats fields. recent_max is a slow/sticky
   * upper envelope of residual_echo_likelihood; divergent_filter_fraction is
   * the fraction of time the AEC adaptive filter looked divergent (rises during
   * double-talk / delay changes). Diagnostic / supplementary use. */
  int32_t has_residual_echo_likelihood_recent_max;
  double  residual_echo_likelihood_recent_max;

  int32_t has_divergent_filter_fraction;
  double  divergent_filter_fraction;
} ApmStats;

/* Returns APM_ABI_VERSION compiled into this binary. */
APM_API int32_t APM_CALL apm_abi_version(void);

/* Returns the WebRTC release string this shim was built against (e.g.
 * "m151.0.7922.138"). Never null; valid for the lifetime of the library. Diagnostic
 * use only. */
APM_API const char* APM_CALL apm_webrtc_version(void);

/*
 * Creates an APM instance from `config`. On success writes an opaque handle to
 * *out_handle and returns APM_OK; the caller owns it and must release it with
 * apm_destroy(). On failure returns a negative status and sets *out_handle to
 * NULL.
 */
APM_API int32_t APM_CALL apm_create(const ApmConfig* config, void** out_handle);

/*
 * Creates an APM instance and injects WebRTC's neural residual echo estimator.
 * `model_data` must contain a complete TFLite model and `model_size` must be
 * greater than zero. The shim copies the bytes during this call; the caller may
 * release its buffer when the function returns. The copied model remains owned
 * by the handle until apm_destroy(). Returns APM_ERR_MODEL when TFLite rejects
 * the model or the estimator cannot be initialized.
 */
APM_API int32_t APM_CALL apm_create_with_neural_residual_echo_model(
    const ApmConfig* config,
    const uint8_t* model_data,
    uint64_t model_size,
    void** out_handle);

/* Destroys a handle from apm_create(). NULL is a no-op. */
APM_API void APM_CALL apm_destroy(void* handle);

/* Updates the render->capture delay hint (ms) at run time. */
APM_API int32_t APM_CALL apm_set_stream_delay_ms(void* handle, int32_t delay_ms);

/*
 * Processes one reverse (render / far-end) chunk. Must be called for the chunk
 * time-aligned with the capture chunk passed to apm_process_capture(), and
 * BEFORE it. `channels` points to `num_channels` buffers of `num_frames`
 * floats each; the data is analyzed (used as the AEC reference) and may be
 * written back in place.
 */
APM_API int32_t APM_CALL apm_process_reverse(void* handle,
                                             const float* const* channels,
                                             int32_t num_channels,
                                             int32_t num_frames);

/*
 * Processes one capture (near-end / mic) chunk in place: HPF -> AEC3 -> NS ->
 * AGC2 are applied and the cleaned signal is written back to the same buffers.
 * Same buffer-shape contract as apm_process_reverse().
 */
APM_API int32_t APM_CALL apm_process_capture(void* handle,
                                             const float* const* channels,
                                             int32_t num_channels,
                                             int32_t num_frames);

/*
 * Fills *stats with the latest metrics. Unlike the process/set functions this
 * may be called from a different thread than the audio callback (APM locks
 * internally for stats). Returns APM_OK on success.
 */
APM_API int32_t APM_CALL apm_get_stats(void* handle, ApmStats* stats);

/*
 * Re-initializes internal state (adaptive filters, buffers, noise estimate)
 * while retaining the configuration. Call when the audio stream stops/restarts.
 * Must not run concurrently with apm_process_*.
 */
APM_API int32_t APM_CALL apm_reset(void* handle);

/* ===========================================================================
 * Standalone AEC3 delay estimator (ABI v3).
 *
 * Exposes WebRTC's AEC3 matched-filter echo-path delay estimator
 * (webrtc::EchoPathDelayEstimator) as a self-contained, time-domain C API,
 * independent of the full APM pipeline. Internally it owns an
 * EchoCanceller3Config + ApmDataDumper + RenderDelayBuffer + the estimator and
 * the render/capture frame blockers, so the caller just feeds time-aligned
 * render/capture samples and reads back a delay in full-rate samples.
 *
 * The handle is NOT an ApmHandle and must be released with apm_delay_destroy().
 * It is single-threaded: do not call apm_delay_* concurrently on one handle.
 *
 * M3 scope: 16000 Hz, mono only (matches the DTLN-aec consumer). Other rates /
 * channel counts return APM_ERR_BAD_ARG from apm_delay_create().
 * =========================================================================== */

/*
 * Creates a delay-estimator instance. On success writes an opaque handle to
 * *out_handle and returns APM_OK. sample_rate must be 16000 and num_channels 1
 * in this version; anything else returns APM_ERR_BAD_ARG.
 */
APM_API int32_t APM_CALL apm_delay_create(int32_t sample_rate,
                                          int32_t num_channels,
                                          void** out_handle);

/* Destroys a handle from apm_delay_create(). NULL is a no-op. */
APM_API void APM_CALL apm_delay_destroy(void* handle);

/*
 * Feeds one chunk of time-aligned render (far-end / playout reference) and
 * capture (near-end / mic) samples, `num_samples` each (mono, interleaving is
 * irrelevant at 1 channel). The samples are internally re-blocked to AEC3's
 * 64-sample blocks, so num_samples may be any positive length.
 *
 * If a new delay estimate became available during this call, writes the
 * estimated render->capture delay in full-rate samples (>= 0, already
 * compensated for the internal DelayEstimatorOffset) to *out_delay_samples,
 * writes 0 (coarse) or 1 (refined) to *out_quality, and returns 1.
 * Returns 0 when no new estimate was produced. Returns a negative status on
 * error. out_delay_samples / out_quality may be NULL if not needed.
 */
APM_API int32_t APM_CALL apm_delay_process(void* handle,
                                           const float* render,
                                           const float* capture,
                                           int32_t num_samples,
                                           int32_t* out_delay_samples,
                                           int32_t* out_quality);

/*
 * Re-initializes the estimator state (matched filters, buffers, accumulators)
 * while keeping the configuration. Call when the audio stream stops/restarts.
 */
APM_API int32_t APM_CALL apm_delay_reset(void* handle);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* VOICEBOX_WEBRTC_APM_SHIM_H_ */
