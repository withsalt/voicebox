# Render-aware DTD v5 training changes

Status: **training not started**. These changes prepare a validation-only ablation round; they do not replace the packaged ONNX model.

## Failure evidence addressed

- 52/88 acoustic false negatives never reached the packaged 0.55 threshold.
- 35 additional false negatives had sparse, echo-vetoed, or fragmented temporal support.
- 54/61 far-end false positives were in the runtime policy's low-correlation blind region and already carried sustained high DTD scores.
- All 18 detected latency failures waited for stable model evidence before the fixed confirmation duration was satisfied.

The corrective training objectives therefore operate on model probabilities and stream events. No benchmark audio, case ID, device ID, device class, hardware delay, or device-specific threshold is used.

## Implemented semantics

- Weighted frame BCE remains the base objective and keeps the locked `positive_weight`.
- Hard-negative loss penalizes both the top-scoring far-only frames and persistent smoothed far-only evidence.
- Onset loss weights only valid frames after a real target onset; it never rewards an oracle pre-trigger.
- Event loss is a differentiable approximation of the production 3-hop smoothing, 180 ms evidence in a 360 ms window, and 80 ms recent-tail requirement.
- Reset loss reruns the first valid post-gap audio from a cleared recurrent state. Removed gap samples are not presented as zero-render inference frames.
- Validation reports cluster-level recall/Lower95, specificity, false triggers/hour with the conservative maximum of Poisson and cluster-aware Upper95, early triggers, right censoring, cluster-worst P50/P95 latency, and errors without flattening streams.
- Validation hardening has separate locked parameters, so changing a training augmentation during an ablation cannot silently change the validation distribution.
- Event validation and saved checkpoints use the frozen 0.55 production threshold; the independently selected frame-F1 threshold is diagnostic only.
- `select_v5.py` accepts either one validation summary or a complete `history.json`, first rejects every frame/event regression (including recall) for every epoch, then ranks survivors lexicographically; it does not hide regressions in a weighted score or discard event-strong epochs behind the legacy `best.pt` frame score.

The existing `candidate-12.json` remains a historical frozen record. Its hash check is expected to reject the modified trainer instead of silently treating this code as candidate-12. Locked values and preregistered new-parameter ranges are in `configs/v5-parameter-lock.json`; the unexecuted experiment order is in `configs/v5-ablation-plan.json`.
