export interface ModelMetadata {
  keyword: string;
  keyword_token_ids: number[];
  keyword_oov_chars: string[];
  sample_rate: number;
  fbank: {
    num_mel_bins: number;
    frame_length_ms: number;
    frame_shift_ms: number;
    window: string;
    dither: number;
    snip_edges: boolean;
    waveform_scale: number;
  };
  lfr: { m: number; n: number };
  cmvn: { add_shift: number[]; rescale: number[] };
  model: {
    input_dim: number;
    output_dim: number;
    num_layers: number;
    proj_dim: number;
    left_order: number;
    right_order: number;
    cache_len: number;
    output_delay_frames: number;
    warmup_frames: number;
    blank_id: number;
  };
}

export function parseMetadata(
  bytes: ArrayBuffer,
  keyword: string,
): ModelMetadata {
  const meta: ModelMetadata = JSON.parse(new TextDecoder().decode(bytes));
  const f = meta?.fbank,
    m = meta?.model;
  if (
    meta.keyword !== keyword ||
    meta.sample_rate !== 16000 ||
    !Array.isArray(meta.keyword_token_ids) ||
    meta.keyword_token_ids.length < 1 ||
    !meta.keyword_token_ids.every(
      (id) => Number.isInteger(id) && id > 0 && id < 2599,
    ) ||
    !Array.isArray(meta.keyword_oov_chars) ||
    meta.keyword_oov_chars.length !== 0 ||
    f?.num_mel_bins !== 80 ||
    f.frame_length_ms !== 25 ||
    f.frame_shift_ms !== 10 ||
    f.window !== "hamming" ||
    f.dither !== 0 ||
    !f.snip_edges ||
    f.waveform_scale !== 32768 ||
    meta.lfr?.m !== 5 ||
    meta.lfr.n !== 3 ||
    m?.input_dim !== 400 ||
    m.output_dim !== 2599 ||
    m.num_layers !== 4 ||
    m.proj_dim !== 128 ||
    m.cache_len !== 11 ||
    m.left_order !== 10 ||
    m.right_order !== 2 ||
    m.output_delay_frames !== 8 ||
    m.warmup_frames !== 35 ||
    m.blank_id !== 0
  ) {
    throw new Error("模型配置与当前音频引擎不兼容，请更新页面后重试。");
  }
  for (const key of ["add_shift", "rescale"] as const) {
    const values = meta.cmvn?.[key];
    if (
      !Array.isArray(values) ||
      values.length !== 400 ||
      !values.every(Number.isFinite)
    ) {
      throw new Error("模型的 CMVN 参数不完整，请清除模型缓存后重试。");
    }
  }
  return meta;
}
