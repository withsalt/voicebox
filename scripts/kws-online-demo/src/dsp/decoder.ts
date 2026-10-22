interface TokenNode {
  token: number;
  frame: number;
  prob: number;
}
interface Hypothesis {
  prefix: number[];
  pb: number;
  pnb: number;
  nodes: TokenNode[];
}
export interface Detection {
  score: number;
  startFrame: number;
  endFrame: number;
}

/** CTC prefix beam search, including repeated-token blank handling and bounded history. */
export class KeywordDecoder {
  private hypotheses: Hypothesis[] = [];
  private frame = 0;
  private readonly allowed: Set<number>;

  constructor(
    private readonly keyword: number[],
    public threshold = 0.5,
    private readonly maxSpan = 100,
  ) {
    this.allowed = new Set([0, ...keyword]);
    this.reset();
  }

  reset() {
    this.hypotheses = [{ prefix: [], pb: 1, pnb: 0, nodes: [] }];
  }

  step(probs: Float32Array): Detection | null {
    const time = this.frame++;
    const top = [-1, -1, -1],
      values = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < probs.length; i++) {
      if (probs[i] <= values[2]) continue;
      let pos = 2;
      while (pos > 0 && probs[i] > values[pos - 1]) pos--;
      for (let j = 2; j > pos; j--) {
        top[j] = top[j - 1];
        values[j] = values[j - 1];
      }
      top[pos] = i;
      values[pos] = probs[i];
    }
    const candidates = top.filter(
      (token, i) => values[i] > 0.05 && this.allowed.has(token),
    );
    if (candidates.length) {
      const next = new Map<string, Hypothesis>();
      const get = (prefix: number[]) => {
        const key = prefix.join(",");
        let hyp = next.get(key);
        if (!hyp) {
          hyp = { prefix, pb: 0, pnb: 0, nodes: [] };
          next.set(key, hyp);
        }
        return hyp;
      };
      for (const token of candidates) {
        const probability = probs[token];
        for (const hyp of this.hypotheses) {
          const last = hyp.prefix.at(-1);
          if (token === 0) {
            const n = get(hyp.prefix);
            n.pb += (hyp.pb + hyp.pnb) * probability;
            n.nodes = hyp.nodes.slice();
          } else if (token === last) {
            if (hyp.pnb > 1e-6) {
              const n = get(hyp.prefix),
                nodes = hyp.nodes.slice();
              if (probability > nodes.at(-1)!.prob)
                nodes[nodes.length - 1] = {
                  token,
                  frame: time,
                  prob: probability,
                };
              n.pnb += hyp.pnb * probability;
              n.nodes = nodes;
            }
            if (hyp.pb > 1e-6) {
              const n = get([...hyp.prefix, token]);
              n.pnb += hyp.pb * probability;
              n.nodes = [
                ...hyp.nodes,
                { token, frame: time, prob: probability },
              ];
            }
          } else {
            const n = get([...hyp.prefix, token]);
            if (!n.nodes.length)
              n.nodes = [
                ...hyp.nodes,
                { token, frame: time, prob: probability },
              ];
            else if (probability > n.nodes.at(-1)!.prob)
              n.nodes[n.nodes.length - 1] = {
                token,
                frame: time,
                prob: probability,
              };
            n.pnb += (hyp.pb + hyp.pnb) * probability;
          }
        }
      }
      this.hypotheses = [...next.values()]
        .sort((a, b) => b.pb + b.pnb - a.pb - a.pnb)
        .slice(0, 20);
    }

    const min = time - this.maxSpan;
    this.hypotheses = this.hypotheses.filter(
      (h) => !h.nodes.length || h.nodes.at(-1)!.frame >= min,
    );
    let best = 0;
    for (const hyp of this.hypotheses) {
      let remove = 0;
      while (hyp.nodes[remove]?.frame < min) remove++;
      if (remove) {
        hyp.nodes = hyp.nodes.slice(remove);
        hyp.prefix = hyp.prefix.slice(remove);
      }
      best = Math.max(best, hyp.pb + hyp.pnb);
    }
    if (!this.hypotheses.some((h) => !h.prefix.length))
      this.hypotheses.push({ prefix: [], nodes: [], pb: best || 1, pnb: 0 });
    if (best > 0 && (best < 1e-10 || best > 1e10))
      for (const hyp of this.hypotheses) {
        hyp.pb /= best;
        hyp.pnb /= best;
      }

    let detection: Detection | null = null;
    for (const hyp of this.hypotheses) {
      for (
        let start = 0;
        start <= hyp.prefix.length - this.keyword.length;
        start++
      ) {
        if (!this.keyword.every((token, i) => token === hyp.prefix[start + i]))
          continue;
        const nodes = hyp.nodes.slice(start, start + this.keyword.length);
        if (nodes.at(-1)!.frame - nodes[0].frame > this.maxSpan) continue;
        const score = Math.sqrt(
          nodes.reduce((product, node) => product * node.prob, 1),
        );
        if (
          score >= this.threshold &&
          (!detection || score > detection.score)
        ) {
          detection = {
            score,
            startFrame: nodes[0].frame,
            endFrame: nodes.at(-1)!.frame,
          };
        }
      }
    }
    return detection;
  }
}
