/*
 * webrtc-apm — C ABI shim implementation. See apm_shim.h for the contract.
 *
 * Wraps one webrtc::AudioProcessing built via BuiltinAudioProcessingBuilder.
 * The double-talk tuning is injected through EchoCanceller3Config (a separate
 * struct from AudioProcessing::Config) via SetEchoCancellerConfig() — applying
 * it through ApplyConfig would silently have no effect.
 */
#include "apm_shim.h"

#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "api/audio/audio_processing.h"
#include "api/audio/audio_processing_statistics.h"
#include "api/audio/builtin_audio_processing_builder.h"
#include "api/audio/echo_canceller3_config.h"
#include "api/audio/neural_residual_echo_estimator_creator.h"
#include "api/environment/environment.h"
#include "api/environment/environment_factory.h"
#include "api/scoped_refptr.h"
#include "third_party/tflite/src/tensorflow/lite/core/kernels/builtin_op_kernels.h"
#include "third_party/tflite/src/tensorflow/lite/model_builder.h"
#include "third_party/tflite/src/tensorflow/lite/mutable_op_resolver.h"

namespace {

// Largest channel count we accept across the C boundary. M1 is mono; the small
// fixed cap lets us build the deinterleaved pointer array on the stack without
// allocating on the real-time path, and rejects absurd inputs early.
constexpr int kMaxChannels = 8;

// Opaque handle backing `void*` in the C API. Owns the APM instance and caches
// the immutable per-stream config so the hot path allocates nothing.
struct ApmHandle {
  // Declaration order is intentional. Members are destroyed in reverse order,
  // so APM (which owns the estimator) dies before the FlatBufferModel, and the
  // model dies before the byte storage backing BuildFromBuffer().
  std::vector<uint8_t> neural_model_data;
  std::unique_ptr<tflite::FlatBufferModel> neural_model;
  webrtc::scoped_refptr<webrtc::AudioProcessing> apm;
  webrtc::StreamConfig stream_config;
  int sample_rate = 0;
  int num_channels = 0;
  int num_frames = 0;  // samples per channel per chunk = sample_rate / 100
};

webrtc::AudioProcessing::Config::NoiseSuppression::Level MapNsLevel(int level) {
  using NS = webrtc::AudioProcessing::Config::NoiseSuppression;
  switch (level) {
    case APM_NS_LOW:
      return NS::kLow;
    case APM_NS_MODERATE:
      return NS::kModerate;
    case APM_NS_VERY_HIGH:
      return NS::kVeryHigh;
    case APM_NS_HIGH:
    default:
      return NS::kHigh;
  }
}

// Validates buffer geometry against what the handle was created for and builds
// a stack array of channel pointers for the APM float interface. Returns APM_OK
// or a negative status; on success `out` holds `num_channels` valid pointers.
int PrepareChannels(const ApmHandle* h,
                    const float* const* channels,
                    int num_channels,
                    int num_frames,
                    const float** out) {
  if (channels == nullptr) {
    return APM_ERR_NULL_ARG;
  }
  if (num_channels != h->num_channels || num_frames != h->num_frames) {
    return APM_ERR_BAD_ARG;
  }
  for (int c = 0; c < num_channels; ++c) {
    if (channels[c] == nullptr) {
      return APM_ERR_NULL_ARG;
    }
    out[c] = channels[c];
  }
  return APM_OK;
}

void RegisterNeuralResidualEchoModelOps(
    tflite::MutableOpResolver* resolver) {
  using namespace tflite::ops::builtin;
  resolver->AddBuiltin(tflite::BuiltinOperator_SLICE, Register_SLICE());
  resolver->AddBuiltin(tflite::BuiltinOperator_RESHAPE, Register_RESHAPE());
  resolver->AddBuiltin(tflite::BuiltinOperator_CONCATENATION,
                       Register_CONCATENATION());
  resolver->AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                       Register_FULLY_CONNECTED(), 1, 12);
  resolver->AddBuiltin(tflite::BuiltinOperator_ADD, Register_ADD());
  resolver->AddBuiltin(tflite::BuiltinOperator_LOGISTIC, Register_LOGISTIC());
  resolver->AddBuiltin(tflite::BuiltinOperator_TANH, Register_TANH());
  resolver->AddBuiltin(tflite::BuiltinOperator_MUL, Register_MUL());
  resolver->AddBuiltin(tflite::BuiltinOperator_SUB, Register_SUB());
  resolver->AddBuiltin(tflite::BuiltinOperator_MAXIMUM, Register_MAXIMUM());
  resolver->AddBuiltin(tflite::BuiltinOperator_MINIMUM, Register_MINIMUM());
}

int CreateApm(const ApmConfig* config,
              const uint8_t* model_data,
              uint64_t model_size,
              void** out_handle) {
  if (out_handle == nullptr) {
    return APM_ERR_NULL_ARG;
  }
  *out_handle = nullptr;
  if (config == nullptr) {
    return APM_ERR_NULL_ARG;
  }
  if (config->struct_size != sizeof(ApmConfig)) {
    return APM_ERR_ABI;
  }
  if (config->num_channels < 1 || config->num_channels > kMaxChannels) {
    return APM_ERR_BAD_ARG;
  }
  switch (config->sample_rate) {
    case 8000:
    case 16000:
    case 32000:
    case 48000:
      break;
    default:
      return APM_ERR_BAD_ARG;
  }

  const bool use_neural_estimator = model_data != nullptr || model_size != 0;
  if (use_neural_estimator &&
      (model_data == nullptr || model_size == 0 ||
       model_size > std::numeric_limits<size_t>::max())) {
    return APM_ERR_BAD_ARG;
  }
  if (use_neural_estimator && config->enable_echo_canceller == 0) {
    return APM_ERR_BAD_ARG;
  }

  try {
    using webrtc::AudioProcessing;

    AudioProcessing::Config apm_config;
    apm_config.high_pass_filter.enabled = config->enable_high_pass_filter != 0;
    // In M151 AEC3 is the only echo canceller (no mobile_mode toggle); enabling
    // echo_canceller selects desktop AEC3.
    apm_config.echo_canceller.enabled = config->enable_echo_canceller != 0;
    apm_config.noise_suppression.enabled =
        config->enable_noise_suppression != 0;
    apm_config.noise_suppression.level = MapNsLevel(config->ns_level);
    // AGC1 stays off; AGC2 is the modern replacement.
    apm_config.gain_controller1.enabled = false;
    apm_config.gain_controller2.enabled = config->enable_agc2 != 0;
    apm_config.gain_controller2.adaptive_digital.enabled =
        config->agc2_adaptive_digital != 0;

    // Double-talk tuning lives in EchoCanceller3Config and must be injected via
    // the builder before Build(). Start from defaults, override the knobs.
    webrtc::EchoCanceller3Config aec3;
    aec3.suppressor.dominant_nearend_detection.enr_threshold =
        config->dt_enr_threshold;
    aec3.suppressor.dominant_nearend_detection.enr_exit_threshold =
        config->dt_enr_exit_threshold;
    aec3.suppressor.dominant_nearend_detection.snr_threshold =
        config->dt_snr_threshold;
    aec3.ep_strength.nearend_len = config->ep_nearend_len;
    webrtc::EchoCanceller3Config::Validate(&aec3);

    auto h = std::make_unique<ApmHandle>();
    webrtc::BuiltinAudioProcessingBuilder builder(apm_config);
    builder.SetEchoCancellerConfig(aec3, std::nullopt);

    if (use_neural_estimator) {
      const size_t model_bytes = static_cast<size_t>(model_size);
      h->neural_model_data.assign(model_data, model_data + model_bytes);
      h->neural_model = tflite::FlatBufferModel::VerifyAndBuildFromBuffer(
          reinterpret_cast<const char*>(h->neural_model_data.data()),
          h->neural_model_data.size());
      if (h->neural_model == nullptr) {
        return APM_ERR_MODEL;
      }

      tflite::MutableOpResolver resolver;
      RegisterNeuralResidualEchoModelOps(&resolver);
      auto estimator = webrtc::CreateNeuralResidualEchoEstimator(
          h->neural_model.get(), &resolver);
      if (estimator == nullptr) {
        return APM_ERR_MODEL;
      }
      builder.SetNeuralResidualEchoEstimator(std::move(estimator));
    }

    h->apm = builder.Build(webrtc::CreateEnvironment());
    if (h->apm == nullptr) {
      return APM_ERR_CREATE;
    }

    h->apm->Initialize();
    if (config->stream_delay_ms > 0) {
      h->apm->set_stream_delay_ms(config->stream_delay_ms);
    }

    h->sample_rate = config->sample_rate;
    h->num_channels = config->num_channels;
    h->num_frames = AudioProcessing::GetFrameSize(config->sample_rate);
    h->stream_config =
        webrtc::StreamConfig(config->sample_rate, config->num_channels);

    *out_handle = h.release();
    return APM_OK;
  } catch (const std::exception&) {
    return APM_ERR_INTERNAL;
  } catch (...) {
    return APM_ERR_INTERNAL;
  }
}

}  // namespace

extern "C" {

int32_t APM_CALL apm_abi_version(void) {
  return APM_ABI_VERSION;
}

const char* APM_CALL apm_webrtc_version(void) {
  // Kept in sync with the official WebRTC revision pinned by webrtc.lock.json.
  return "m151.0.7922.138";
}

int32_t APM_CALL apm_create(const ApmConfig* config, void** out_handle) {
  return CreateApm(config, nullptr, 0, out_handle);
}

int32_t APM_CALL apm_create_with_neural_residual_echo_model(
    const ApmConfig* config,
    const uint8_t* model_data,
    uint64_t model_size,
    void** out_handle) {
  if (model_data == nullptr) {
    if (out_handle != nullptr) {
      *out_handle = nullptr;
    }
    return APM_ERR_NULL_ARG;
  }
  return CreateApm(config, model_data, model_size, out_handle);
}

void APM_CALL apm_destroy(void* handle) {
  if (handle == nullptr) {
    return;
  }
  try {
    delete static_cast<ApmHandle*>(handle);
  } catch (...) {
    // Destructors must not throw across the boundary; nothing else to do.
  }
}

int32_t APM_CALL apm_set_stream_delay_ms(void* handle, int32_t delay_ms) {
  if (handle == nullptr) {
    return APM_ERR_NULL_ARG;
  }
  try {
    auto* h = static_cast<ApmHandle*>(handle);
    return h->apm->set_stream_delay_ms(delay_ms);
  } catch (...) {
    return APM_ERR_INTERNAL;
  }
}

int32_t APM_CALL apm_process_reverse(void* handle,
                                     const float* const* channels,
                                     int32_t num_channels,
                                     int32_t num_frames) {
  if (handle == nullptr) {
    return APM_ERR_NULL_ARG;
  }
  try {
    auto* h = static_cast<ApmHandle*>(handle);
    const float* src[kMaxChannels];
    int rc = PrepareChannels(h, channels, num_channels, num_frames, src);
    if (rc != APM_OK) {
      return rc;
    }
    // Render is analyzed and may be written back in place; src == dest.
    float* dest[kMaxChannels];
    for (int c = 0; c < num_channels; ++c) {
      dest[c] = const_cast<float*>(src[c]);
    }
    return h->apm->ProcessReverseStream(src, h->stream_config,
                                        h->stream_config, dest);
  } catch (...) {
    return APM_ERR_INTERNAL;
  }
}

int32_t APM_CALL apm_process_capture(void* handle,
                                     const float* const* channels,
                                     int32_t num_channels,
                                     int32_t num_frames) {
  if (handle == nullptr) {
    return APM_ERR_NULL_ARG;
  }
  try {
    auto* h = static_cast<ApmHandle*>(handle);
    const float* src[kMaxChannels];
    int rc = PrepareChannels(h, channels, num_channels, num_frames, src);
    if (rc != APM_OK) {
      return rc;
    }
    // In-place cleanup: cleaned signal overwrites the input buffers.
    float* dest[kMaxChannels];
    for (int c = 0; c < num_channels; ++c) {
      dest[c] = const_cast<float*>(src[c]);
    }
    return h->apm->ProcessStream(src, h->stream_config, h->stream_config, dest);
  } catch (...) {
    return APM_ERR_INTERNAL;
  }
}

int32_t APM_CALL apm_get_stats(void* handle, ApmStats* stats) {
  if (handle == nullptr || stats == nullptr) {
    return APM_ERR_NULL_ARG;
  }
  try {
    auto* h = static_cast<ApmHandle*>(handle);
    webrtc::AudioProcessingStats s = h->apm->GetStatistics();

    stats->has_echo_return_loss = s.echo_return_loss.has_value() ? 1 : 0;
    stats->echo_return_loss = s.echo_return_loss.value_or(0.0);

    stats->has_echo_return_loss_enhancement =
        s.echo_return_loss_enhancement.has_value() ? 1 : 0;
    stats->echo_return_loss_enhancement =
        s.echo_return_loss_enhancement.value_or(0.0);

    stats->has_delay_ms = s.delay_ms.has_value() ? 1 : 0;
    stats->delay_ms = s.delay_ms.value_or(0);

    stats->has_delay_median_ms = s.delay_median_ms.has_value() ? 1 : 0;
    stats->delay_median_ms = s.delay_median_ms.value_or(0);

    stats->has_residual_echo_likelihood =
        s.residual_echo_likelihood.has_value() ? 1 : 0;
    stats->residual_echo_likelihood =
        s.residual_echo_likelihood.value_or(0.0);

    /* ABI v2 additions. */
    stats->has_residual_echo_likelihood_recent_max =
        s.residual_echo_likelihood_recent_max.has_value() ? 1 : 0;
    stats->residual_echo_likelihood_recent_max =
        s.residual_echo_likelihood_recent_max.value_or(0.0);

    stats->has_divergent_filter_fraction =
        s.divergent_filter_fraction.has_value() ? 1 : 0;
    stats->divergent_filter_fraction =
        s.divergent_filter_fraction.value_or(0.0);

    return APM_OK;
  } catch (...) {
    return APM_ERR_INTERNAL;
  }
}

int32_t APM_CALL apm_reset(void* handle) {
  if (handle == nullptr) {
    return APM_ERR_NULL_ARG;
  }
  try {
    auto* h = static_cast<ApmHandle*>(handle);
    return h->apm->Initialize();
  } catch (...) {
    return APM_ERR_INTERNAL;
  }
}

}  // extern "C"
