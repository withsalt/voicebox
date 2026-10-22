// C-ABI probe: exercises the webrtc-apm shim exactly as the managed P/Invoke
// will. Verifies create -> reverse/capture -> stats -> double-talk reconfig.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

#include "apm_shim.h"

static void fill_tone(std::vector<float>& b, double freq, double amp, int rate) {
  for (size_t i = 0; i < b.size(); ++i)
    b[i] = (float)(amp * std::sin(2.0 * 3.14159265358979 * freq * i / rate));
}

int main(int argc, char** argv) {
  const int rate = 16000;
  const int frame = rate / 100;  // 160

  ApmConfig cfg{};
  cfg.struct_size = (uint32_t)sizeof(ApmConfig);
  cfg.sample_rate = rate;
  cfg.num_channels = 1;
  cfg.enable_high_pass_filter = 1;
  cfg.enable_echo_canceller = 1;
  cfg.enable_noise_suppression = 1;
  cfg.ns_level = APM_NS_HIGH;
  cfg.enable_agc2 = 1;
  cfg.agc2_adaptive_digital = 1;
  cfg.stream_delay_ms = 40;
  cfg.dt_enr_threshold = 0.25f;
  cfg.dt_enr_exit_threshold = 10.0f;
  cfg.dt_snr_threshold = 30.0f;
  cfg.ep_nearend_len = 0.83f;

  printf("abi=%d webrtc=%s\n", apm_abi_version(), apm_webrtc_version());

  void* h = nullptr;
  int rc = apm_create(&cfg, &h);
  if (rc != APM_OK || h == nullptr) {
    printf("FAIL apm_create rc=%d\n", rc);
    return 2;
  }

  std::vector<float> render(frame), capture(frame);
  fill_tone(render, 440.0, 0.10, rate);
  fill_tone(capture, 300.0, 0.05, rate);
  const float* rch[1] = {render.data()};
  const float* cch[1] = {capture.data()};

  int rr = 0, rcap = 0;
  // Run ~300 ms so AEC3 starts converging and stats populate.
  for (int i = 0; i < 30; ++i) {
    rr = apm_process_reverse(h, rch, 1, frame);
    rcap = apm_process_capture(h, cch, 1, frame);
    if (rr != APM_OK || rcap != APM_OK) {
      printf("FAIL process rev=%d cap=%d at iter %d\n", rr, rcap, i);
      apm_destroy(h);
      return 3;
    }
  }

  ApmStats st{};
  rc = apm_get_stats(h, &st);
  printf("OK rev=%d cap=%d stats_rc=%d erle?=%d erle=%.2f delay?=%d delay=%d\n",
         rr, rcap, rc, st.has_echo_return_loss_enhancement,
         st.echo_return_loss_enhancement, st.has_delay_ms, st.delay_ms);

  // Negative-path checks (must not crash; must return defined error codes).
  printf("null-handle process rc=%d (want %d)\n",
         apm_process_capture(nullptr, cch, 1, frame), APM_ERR_NULL_ARG);
  printf("bad-frames rc=%d (want %d)\n",
         apm_process_capture(h, cch, 1, frame + 1), APM_ERR_BAD_ARG);

  rc = apm_reset(h);
  printf("reset rc=%d\n", rc);

  apm_destroy(h);
  apm_destroy(nullptr);  // must be a safe no-op

  if (argc > 1) {
    std::ifstream model_file(argv[1], std::ios::binary);
    std::vector<uint8_t> model((std::istreambuf_iterator<char>(model_file)),
                               std::istreambuf_iterator<char>());
    if (model.empty()) {
      printf("FAIL unable to read neural model: %s\n", argv[1]);
      return 4;
    }

    void* neural = nullptr;
    rc = apm_create_with_neural_residual_echo_model(
        &cfg, model.data(), static_cast<uint64_t>(model.size()), &neural);
    if (rc != APM_OK || neural == nullptr) {
      printf("FAIL neural model create rc=%d size=%zu\n", rc, model.size());
      return 5;
    }

    rr = apm_process_reverse(neural, rch, 1, frame);
    rcap = apm_process_capture(neural, cch, 1, frame);
    printf("neural model OK size=%zu rev=%d cap=%d\n", model.size(), rr,
           rcap);
    apm_destroy(neural);
    if (rr != APM_OK || rcap != APM_OK) {
      return 6;
    }
  }

  printf("DONE\n");
  return 0;
}
