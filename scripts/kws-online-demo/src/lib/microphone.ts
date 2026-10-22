export class Microphone {
  private context: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private node: AudioWorkletNode | null = null;
  private stopped = false;

  async start(onSamples: (samples: Float32Array) => void) {
    if (!isSecureContext || !navigator.mediaDevices?.getUserMedia)
      throw new Error("录音需要 HTTPS 或 localhost。请通过安全地址打开页面。");
    this.context = new AudioContext({ latencyHint: "interactive" });
    await this.context.resume();
    if (!this.context.audioWorklet)
      throw new Error(
        "浏览器不支持 AudioWorklet，请使用最新版 Chrome 或 Edge。",
      );
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: { ideal: 1 },
        sampleRate: { ideal: 16000 },
        echoCancellation: true,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
    if (this.stopped) {
      await this.stop();
      throw new Error("录音已取消。");
    }
    await this.context.audioWorklet.addModule(
      new URL("audio-capture.js", document.baseURI).href,
    );
    if (this.stopped) throw new Error("录音已取消。");
    this.source = this.context.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.context, "voicebox-capture", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });
    this.node.port.onmessage = ({ data }: MessageEvent<Float32Array>) => {
      if (!this.stopped) onSamples(data);
    };
    // Connect only after the engine has been reset for the actual device rate.
    return {
      sampleRate: this.context.sampleRate,
      settings: this.stream.getAudioTracks()[0].getSettings(),
    };
  }

  connect(onEnded: () => void) {
    if (
      this.stopped ||
      !this.context ||
      !this.node ||
      !this.source ||
      !this.stream
    )
      return;
    this.stream.getAudioTracks()[0].onended = onEnded;
    this.context.onstatechange = () => {
      if (
        !this.stopped &&
        (this.context?.state === "suspended" ||
          this.context?.state === "closed")
      )
        onEnded();
    };
    this.source.connect(this.node);
    this.node.connect(this.context.destination);
  }

  async stop() {
    this.stopped = true;
    this.stream?.getTracks().forEach((track) => {
      track.onended = null;
      track.stop();
    });
    this.stream = null;
    this.node?.disconnect();
    this.source?.disconnect();
    if (this.node) this.node.port.onmessage = null;
    this.node = null;
    this.source = null;
    const context = this.context;
    this.context = null;
    if (context && context.state !== "closed")
      await context.close().catch(() => {});
  }
}
