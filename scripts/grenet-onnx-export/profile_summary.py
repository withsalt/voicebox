"""Stream an ONNX Runtime Chrome trace and summarize operator/node latency."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def summarize(path):
    operators = defaultdict(lambda: [0, 0])
    nodes = defaultdict(lambda: [0, 0])
    providers = defaultdict(lambda: [0, 0])
    runs = []
    with path.open("r", encoding="utf-8") as stream:
        for raw in stream:
            line = raw.strip().rstrip(",")
            if not line or line in "[]":
                continue
            event = json.loads(line)
            duration = event.get("dur", 0)
            if event.get("cat") == "Session" and event.get("name") == "model_run":
                runs.append(duration)
            args = event.get("args", {})
            if "op_name" not in args:
                continue
            operator = args.get("op_name", "unknown")
            provider = args.get("provider", "unknown")
            name = event.get("name", "unknown")
            for key, group in ((operator, operators), (name, nodes), (provider, providers)):
                group[key][0] += duration
                group[key][1] += 1
    return operators, nodes, providers, runs


def rows(group, total, limit):
    for name, (duration, count) in sorted(group.items(), key=lambda item: item[1][0], reverse=True)[:limit]:
        yield {
            "name": name,
            "calls": count,
            "total_ms": round(duration / 1000, 3),
            "mean_us": round(duration / count, 3),
            "percent": round(100 * duration / total, 2) if total else 0,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    operators, nodes, providers, runs = summarize(args.profile)
    total = sum(duration for duration, _ in operators.values())
    result = {
        "profile": str(args.profile),
        "kernel_events": sum(count for _, count in operators.values()),
        "kernel_total_ms": round(total / 1000, 3),
        "model_runs": len(runs),
        "model_run_mean_ms": round(sum(runs) / len(runs) / 1000, 3) if runs else None,
        "providers": list(rows(providers, total, args.top)),
        "operators": list(rows(operators, total, args.top)),
        "nodes": list(rows(nodes, total, args.top)),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
