"""Local visualizer for the tensor/operator graphs in artifacts/data."""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from collections import Counter, deque
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "artifacts" / "data"
HTML_FILE = Path(__file__).with_name("graph_visualizer.html")


@lru_cache(maxsize=100)
def load_case(case_name: str) -> dict:
    safe = Path(case_name).name
    if not safe.startswith("case_") or not safe.endswith(".json"):
        raise ValueError("invalid case name")
    path = DATA_DIR / safe
    if not path.is_file():
        raise FileNotFoundError(safe)
    data = json.loads(path.read_text(encoding="utf-8"))
    ops = {str(x["id"]): x for x in data["ops"]}
    tensors = {str(x["id"]): x for x in data["tensors"]}
    neighbors: dict[str, list[str]] = {f"o:{key}": [] for key in ops}
    neighbors.update({f"t:{key}": [] for key in tensors})
    for edge in data["edges"]:
        source = str(edge["source"])
        target = str(edge["target"])
        source_key = f"o:{source}" if source in ops else f"t:{source}"
        target_key = f"o:{target}" if target in ops else f"t:{target}"
        neighbors.setdefault(source_key, []).append(target_key)
        neighbors.setdefault(target_key, []).append(source_key)
    return {"raw": data, "ops": ops, "tensors": tensors, "neighbors": neighbors}


def critical_path(ops: dict[str, dict], edges: list[dict]) -> tuple[int, list[str]]:
    """Longest weighted path over operation dependencies, including COPY ops."""
    incoming = {f"o:{key}": [] for key in ops}
    producer: dict[str, str] = {}
    consumers: dict[str, list[str]] = {}
    for edge in edges:
        source, target = str(edge["source"]), str(edge["target"])
        if source in ops and target not in ops:
            producer[target] = source
        elif source not in ops and target in ops:
            consumers.setdefault(source, []).append(target)
    for tensor, consumer_ids in consumers.items():
        if tensor in producer:
            for consumer in consumer_ids:
                if producer[tensor] != consumer:
                    incoming[f"o:{consumer}"].append(f"o:{producer[tensor]}")
    indegree = {key: 0 for key in incoming}
    outgoing = {key: [] for key in incoming}
    for target, sources in incoming.items():
        for source in set(sources):
            outgoing[source].append(target)
            indegree[target] += 1
    queue = deque(key for key, degree in indegree.items() if degree == 0)
    distance = {key: int(ops[key[2:]].get("cycles", 0)) for key in incoming}
    previous: dict[str, str | None] = {key: None for key in incoming}
    visited = 0
    while queue:
        source = queue.popleft()
        visited += 1
        for target in outgoing[source]:
            candidate = distance[source] + int(ops[target[2:]].get("cycles", 0))
            if candidate > distance[target]:
                distance[target], previous[target] = candidate, source
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if visited != len(incoming):
        return 0, []
    end = max(distance, key=distance.get, default=None)
    path: list[str] = []
    while end:
        path.append(end)
        end = previous[end]
    return max(distance.values(), default=0), list(reversed(path))


def summary(case_name: str) -> dict:
    case = load_case(case_name)
    raw, ops, tensors = case["raw"], case["ops"], case["tensors"]
    path_cycles, path = critical_path(ops, raw["edges"])
    op_counts = Counter(x["op"] for x in raw["ops"])
    pipe_counts = Counter(x["pipe"] for x in raw["ops"])
    storage = Counter()
    storage_bytes = Counter()
    for item in raw["tensors"]:
        storage[item["pos"]] += 1
        storage_bytes[item["pos"]] += int(item["size"])
    focus = max((x for x in raw["ops"] if x["op"] not in {"COPY_IN", "COPY_OUT"}), key=lambda x: x.get("cycles", 0), default=raw["ops"][0])
    return {
        "case": case_name,
        "nodes": len(raw["ops"]) + len(raw["tensors"]),
        "ops": len(raw["ops"]),
        "tensors": len(raw["tensors"]),
        "edges": len(raw["edges"]),
        "total_cycles": sum(int(x.get("cycles", 0)) for x in raw["ops"]),
        "critical_path_cycles": path_cycles,
        "critical_path_length": len(path),
        "critical_path": [key[2:] for key in path],
        "default_focus": str(focus["id"]),
        "op_counts": dict(op_counts),
        "pipe_counts": dict(pipe_counts),
        "storage_counts": dict(storage),
        "storage_bytes": dict(storage_bytes),
        "max_tensor_bytes": max((int(x["size"]) for x in raw["tensors"]), default=0),
    }


def graph_slice(case_name: str, focus: str | None, radius: int, limit: int, kind: str, pipe: str) -> dict:
    case = load_case(case_name)
    raw, ops, tensors, neighbors = case["raw"], case["ops"], case["tensors"], case["neighbors"]
    candidates = [key for key, item in ops.items() if not pipe or item["pipe"] == pipe]
    if not focus or focus not in ops:
        focus = max(candidates or list(ops), key=lambda key: int(ops[key].get("cycles", 0)))
    start = f"o:{focus}"
    selected = {start}
    frontier = deque([(start, 0)])
    while frontier and len(selected) < limit:
        node, depth = frontier.popleft()
        if depth >= max(0, radius):
            continue
        for neighbor in neighbors.get(node, []):
            if neighbor in selected:
                continue
            selected.add(neighbor)
            frontier.append((neighbor, depth + 1))
            if len(selected) >= limit:
                break
    if kind != "all":
        selected = {key for key in selected if (key.startswith("o:") and kind == "ops") or (key.startswith("t:") and kind == "tensors")}
        selected.add(start)
    nodes = []
    for key in selected:
        node_id = key[2:]
        if key.startswith("o:"):
            item = ops[node_id]
            nodes.append({"id": key, "raw_id": node_id, "type": "op", "label": item["op"], "pipe": item["pipe"], "cycles": item.get("cycles", 0), "focus": node_id == focus})
        else:
            item = tensors[node_id]
            nodes.append({"id": key, "raw_id": node_id, "type": "tensor", "label": item["pos"], "pos": item["pos"], "size": item.get("size", 0), "focus": False})
    edges = [{"source": source, "target": target} for source in selected for target in neighbors.get(source, []) if target in selected and source < target]
    return {"case": case_name, "focus": focus, "nodes": nodes, "edges": edges, "truncated": len(selected) >= limit}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                body, content_type = HTML_FILE.read_bytes(), "text/html; charset=utf-8"
            elif parsed.path == "/api/cases":
                body = json.dumps([summary(path.name) for path in sorted(DATA_DIR.glob("case_*.json"))], ensure_ascii=False).encode()
                content_type = "application/json; charset=utf-8"
            elif parsed.path == "/api/case":
                query = parse_qs(parsed.query)
                case = query.get("case", ["case_001.json"])[0]
                body = json.dumps(graph_slice(case, query.get("focus", [None])[0], int(query.get("radius", [2])[0]), min(800, max(20, int(query.get("limit", [240])[0]))), query.get("kind", ["all"])[0], query.get("pipe", [""])[0]), ensure_ascii=False).encode()
                content_type = "application/json; charset=utf-8"
            elif parsed.path == "/api/ops":
                query = parse_qs(parsed.query)
                case = load_case(query.get("case", ["case_001.json"])[0])
                body = json.dumps([{"id": str(item["id"]), "op": item["op"], "pipe": item["pipe"], "cycles": item.get("cycles", 0)} for item in case["raw"]["ops"]], ensure_ascii=False).encode()
                content_type = "application/json; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            self.send_error(400, str(exc))

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize computation graphs in artifacts/data")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Graph visualizer: {url}")
    if not args.no_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
