/*
 * webrtc-apm — standalone AEC3 delay-estimator C ABI (ABI v3). See apm_shim.h.
 *
 * Drives webrtc::EchoPathDelayEstimator (the AEC3 matched-filter echo-path
 * delay estimator) directly as a pure delay sensor, outside the full APM
 * pipeline. For each aligned 64-sample (render, capture) block pair:
 *
 *   render_buffer->Insert(render_block);
 *   render_buffer->PrepareCaptureProcessing();
 *   estimate = estimator->EstimateDelay(
 *                  render_buffer->GetDownsampledRenderBuffer(), capture_block);
 *
 * The estimator reports the total acoustic render→capture delay in full-rate
 * samples (matched-filter lag × down-sampling factor), minus a fixed -128
 * sample bias (calibrated empirically, see kDelayEstimatorOffsetSamples).
 *
 * AEC3 internally operates on FloatS16-scaled samples (~[-32768, 32767]), so
 * caller's [-1, 1] input is scaled by 32768 before feeding the estimator
 * (line 186-187), ensuring activity gates trigger correctly.
 *
 * The estimator consumes raw 64-sample AEC3 blocks (kBlockSize, 1 band at
 * 16 kHz) — the FrameBlocker / 80-sample sub-frame machinery is only needed by
 * the full APM's 10 ms framing. We internally re-block the caller's
 * arbitrary-length input into 64-sample blocks.
 *
 * Scope: 16 kHz mono. Every entry point is noexcept and returns a status code.
 */
#include "apm_shim.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "api/audio/echo_canceller3_config.h"
#include "modules/audio_processing/aec3/aec3_common.h"
#include "modules/audio_processing/aec3/block.h"
#include "modules/audio_processing/aec3/delay_estimate.h"
#include "modules/audio_processing/aec3/echo_path_delay_estimator.h"
#include "modules/audio_processing/aec3/render_delay_buffer.h"
#include "modules/audio_processing/logging/apm_data_dumper.h"

namespace {

// M3 scope. kBlockSize (== 64) and the single-band layout are only valid for
// 16 kHz; supporting >16 kHz would additionally need a band split.
constexpr int kSupportedSampleRate = 16000;
constexpr int kSupportedChannels = 1;
constexpr int kNumBands = 1;  // NumBandsForRate(16000)

// AEC3's internal float pipeline is FloatS16-scaled (samples in ~[-32768,
// 32767]), not the [-1, 1] of the public APM float interface — its activity /
// power gates (e.g. AlignmentMixing::activity_power_threshold == 10000) assume
// that scale. The full APM does this conversion in AudioBuffer; driving the
// AEC3 components directly we must scale the caller's [-1, 1] samples ourselves.
constexpr float kFloatToS16 = 32768.0f;

// Fixed bias between EchoPathDelayEstimator's reported delay and the true
// acoustic render->capture delay. RenderDelayBuffer::DelayEstimatorOffset()
// would give this, but that static is inlined away in the prebuilt lib and not
// externally linkable, so it was calibrated empirically against known synthetic
// delays (see WebRtcAec3DelayTests): the estimator reports (true_delay - 128)
// consistently for delays from 256..3200 samples, i.e. a constant -128 (= 2
// blocks) bias. Subtracting -128 recovers the true delay. Reported = est - this.
constexpr int kDelayEstimatorOffsetSamples = -128;

// Opaque handle backing the void* in the C API. Owns the estimator graph plus a
// small re-blocking accumulator. Single-threaded by contract.
struct DelayHandle {
  explicit DelayHandle(int sample_rate, int num_channels)
      : data_dumper(std::make_unique<webrtc::ApmDataDumper>(0)),
        render_block(kNumBands, num_channels),
        capture_block(kNumBands, num_channels) {
    webrtc::EchoCanceller3Config::Validate(&config);
    render_buffer.reset(webrtc::RenderDelayBuffer::Create(
        config, sample_rate, static_cast<size_t>(num_channels)));
    estimator = std::make_unique<webrtc::EchoPathDelayEstimator>(
        data_dumper.get(), config, static_cast<size_t>(num_channels));
  }

  webrtc::EchoCanceller3Config config;
  std::unique_ptr<webrtc::ApmDataDumper> data_dumper;
  std::unique_ptr<webrtc::RenderDelayBuffer> render_buffer;
  std::unique_ptr<webrtc::EchoPathDelayEstimator> estimator;

  webrtc::Block render_block;
  webrtc::Block capture_block;

  // Leftover (<64) input samples carried between calls so the caller may feed
  // any chunk size. render_acc and capture_acc stay equal length / aligned.
  std::vector<float> render_acc;
  std::vector<float> capture_acc;

  int last_delay = 0;
  int last_quality = 0;
  bool has_delay = false;
};

// Runs one aligned (render, capture) 64-sample block pair through the estimator.
// Returns true if a delay estimate was produced (state updated on the handle).
bool ProcessBlockPair(DelayHandle* h) {
  h->render_buffer->Insert(h->render_block);
  h->render_buffer->PrepareCaptureProcessing();
  std::optional<webrtc::DelayEstimate> est = h->estimator->EstimateDelay(
      h->render_buffer->GetDownsampledRenderBuffer(), h->capture_block);
  if (!est.has_value()) {
    return false;
  }
  int delay = static_cast<int>(est->delay) - kDelayEstimatorOffsetSamples;
  if (delay < 0) {
    delay = 0;
  }
  h->last_delay = delay;
  h->last_quality =
      est->quality == webrtc::DelayEstimate::Quality::kRefined ? 1 : 0;
  h->has_delay = true;
  return true;
}

}  // namespace

extern "C" {

int32_t APM_CALL apm_delay_create(int32_t sample_rate,
                                  int32_t num_channels,
                                  void** out_handle) {
  if (out_handle == nullptr) {
    return APM_ERR_NULL_ARG;
  }
  *out_handle = nullptr;
  if (sample_rate != kSupportedSampleRate ||
      num_channels != kSupportedChannels) {
    return APM_ERR_BAD_ARG;
  }
  try {
    *out_handle = new DelayHandle(sample_rate, num_channels);
    return APM_OK;
  } catch (...) {
    return APM_ERR_INTERNAL;
  }
}

void APM_CALL apm_delay_destroy(void* handle) {
  if (handle == nullptr) {
    return;
  }
  try {
    delete static_cast<DelayHandle*>(handle);
  } catch (...) {
    // Destructors must not throw across the boundary.
  }
}

int32_t APM_CALL apm_delay_process(void* handle,
                                   const float* render,
                                   const float* capture,
                                   int32_t num_samples,
                                   int32_t* out_delay_samples,
                                   int32_t* out_quality) {
  if (handle == nullptr || render == nullptr || capture == nullptr) {
    return APM_ERR_NULL_ARG;
  }
  if (num_samples < 0) {
    return APM_ERR_BAD_ARG;
  }
  try {
    auto* h = static_cast<DelayHandle*>(handle);
    bool produced = false;

    h->render_acc.insert(h->render_acc.end(), render, render + num_samples);
    h->capture_acc.insert(h->capture_acc.end(), capture, capture + num_samples);

    const size_t bs = webrtc::kBlockSize;  // 64
    size_t pos = 0;
    while (h->render_acc.size() - pos >= bs &&
           h->capture_acc.size() - pos >= bs) {
      std::span<float, webrtc::kBlockSize> rv = h->render_block.View(0, 0);
      std::span<float, webrtc::kBlockSize> cv = h->capture_block.View(0, 0);
      const float* rsrc = h->render_acc.data() + pos;
      const float* csrc = h->capture_acc.data() + pos;
      for (size_t i = 0; i < bs; ++i) {
        rv[i] = rsrc[i] * kFloatToS16;  // [-1,1] -> FloatS16 scale for AEC3
        cv[i] = csrc[i] * kFloatToS16;
      }
      pos += bs;
      produced |= ProcessBlockPair(h);
    }

    if (pos > 0) {
      h->render_acc.erase(h->render_acc.begin(), h->render_acc.begin() + pos);
      h->capture_acc.erase(h->capture_acc.begin(),
                           h->capture_acc.begin() + pos);
    }

    if (produced) {
      if (out_delay_samples != nullptr) {
        *out_delay_samples = h->last_delay;
      }
      if (out_quality != nullptr) {
        *out_quality = h->last_quality;
      }
      return 1;
    }
    return 0;
  } catch (...) {
    return APM_ERR_INTERNAL;
  }
}

int32_t APM_CALL apm_delay_reset(void* handle) {
  if (handle == nullptr) {
    return APM_ERR_NULL_ARG;
  }
  try {
    auto* h = static_cast<DelayHandle*>(handle);
    h->render_buffer->Reset();
    h->estimator->Reset(/*reset_delay_confidence=*/true);
    h->render_acc.clear();
    h->capture_acc.clear();
    h->has_delay = false;
    h->last_delay = 0;
    h->last_quality = 0;
    return APM_OK;
  } catch (...) {
    return APM_ERR_INTERNAL;
  }
}

}  // extern "C"
