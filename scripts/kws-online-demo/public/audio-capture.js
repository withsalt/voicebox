// The worklet only captures PCM. DSP and inference run in a separate Worker.
class VoiceBoxCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(Math.round(sampleRate * 0.02));
    this.offset = 0;
  }
  process(inputs) {
    const channels = inputs[0];
    if (!channels?.length) return true;
    for (let i = 0; i < channels[0].length; i++) {
      let sample = 0;
      for (const channel of channels) sample += channel[i] || 0;
      this.buffer[this.offset++] = sample / channels.length;
      if (this.offset === this.buffer.length) {
        this.port.postMessage(this.buffer, [this.buffer.buffer]);
        this.buffer = new Float32Array(Math.round(sampleRate * 0.02));
        this.offset = 0;
      }
    }
    // Outputs remain zero: the microphone is never played through the speaker.
    return true;
  }
}
registerProcessor("voicebox-capture", VoiceBoxCapture);
