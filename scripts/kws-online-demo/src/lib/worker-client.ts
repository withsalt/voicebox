export class WorkerClient {
  private readonly worker = new Worker(
    new URL("../worker/kws.worker.ts", import.meta.url),
    { type: "module" },
  );
  private sequence = 0;
  private readonly requests = new Map<
    number,
    {
      resolve: (value: any) => void;
      reject: (error: Error) => void;
      timer: ReturnType<typeof setTimeout>;
    }
  >();

  constructor() {
    this.worker.onmessage = ({ data }) => {
      const pending = this.requests.get(data.id);
      if (!pending) return;
      clearTimeout(pending.timer);
      this.requests.delete(data.id);
      if (data.error) pending.reject(new Error(data.error));
      else pending.resolve(data.result);
    };
    this.worker.onerror = (event) => {
      this.fail(
        new Error(
          `音频引擎启动失败：${event.message || "请检查浏览器与运行时文件。"}`,
        ),
      );
    };
  }

  call<T = void>(
    type: string,
    payload: unknown = {},
    transfer: Transferable[] = [],
  ): Promise<T> {
    const id = ++this.sequence;
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.requests.delete(id);
        reject(new Error("音频引擎响应超时，请重新加载模型。"));
      }, 60_000);
      this.requests.set(id, { resolve, reject, timer });
      try {
        this.worker.postMessage({ id, type, payload }, transfer);
      } catch (error) {
        clearTimeout(timer);
        this.requests.delete(id);
        reject(error);
      }
    });
  }

  private fail(error: Error) {
    for (const request of this.requests.values()) {
      clearTimeout(request.timer);
      request.reject(error);
    }
    this.requests.clear();
  }

  terminate() {
    this.worker.terminate();
    this.fail(new Error("任务已取消。"));
  }
}
