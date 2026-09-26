# ============================================================
# 人工智能工具信息 | AI Tool Information
#   工具名称 (Tool Name)        : GLM-5.2
#   版本/型号 (Version/Model)   : GLM-5.2
#   开发机构/公司 (Developer)    : 智谱AI (Zhipu AI / zai-org)
#   版本颁布日期 (Release Date) : 2026-06-13
#   声明：本程序及代码是在人工智能工具辅助下完成的
# ============================================================
"""``artifacts/data`` 中张量/算子图的本地浏览器可视化。

启动一个 ``ThreadingHTTPServer``，把 ``graph_visualizer.html`` 与三类 JSON
API 暴露出来：
    * ``/``            —— 返回 HTML 单页应用；
    * ``/api/cases``   —— 全部 case 的概览（含关键路径统计）；
    * ``/api/case``    —— 指定 case 的子图切片（带焦点节点 BFS 邻域）；
    * ``/api/ops``     —— 指定 case 的所有算子（id/op/pipe/cycles）。

除非 ``--no-browser``，启动时会自动用浏览器打开首页。
"""

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
# HTML 单页文件与本脚本同目录；不在代码里嵌 HTML，便于离线修改。
HTML_FILE = Path(__file__).with_name("graph_visualizer.html")


@lru_cache(maxsize=100)
def load_case(case_name: str) -> dict:
    """加载并索引一张 case 图。

    返回结构：
        * ``raw``      —— 原始 JSON；
        * ``ops``/``tensors`` —— key 是 op_id 字符串的索引字典；
        * ``neighbors`` —— ``"o:<id>"``/``"t:<id>"`` 形态的双向邻接表，供
          焦点切片做 BFS 使用。
    安全校验：只接受 ``case_*.json`` 命名的输入，防止路径穿越。
    """
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
    # 同时建立 op↔op、op↔tensor、tensor↔op 双向邻接。
    for edge in data["edges"]:
        source = str(edge["source"])
        target = str(edge["target"])
        source_key = f"o:{source}" if source in ops else f"t:{source}"
        target_key = f"o:{target}" if target in ops else f"t:{target}"
        neighbors.setdefault(source_key, []).append(target_key)
        neighbors.setdefault(target_key, []).append(source_key)
    return {"raw": data, "ops": ops, "tensors": tensors, "neighbors": neighbors}


def critical_path(ops: dict[str, dict], edges: list[dict]) -> tuple[int, list[str]]:
    """最长加权路径（含 COPY 节点）。

    先把 tensor 中转依赖桥接成 op→op 边，再做标准拓扑 DP：
        * distance[v] = ops[v].cycles + max(distance[u] for u in preds[v])；
        * 同时记录 previous[v]，便于回溯路径。
    若检测到环（visited != len(incoming)）返回 (0, [])。
    """
    incoming = {f"o:{key}": [] for key in ops}
    producer: dict[str, str] = {}
    consumers: dict[str, list[str]] = {}
    for edge in edges:
        source, target = str(edge["source"]), str(edge["target"])
        if source in ops and target not in ops:
            producer[target] = source
        elif source not in ops and target in ops:
            consumers.setdefault(source, []).append(target)
    # tensor 既被某个 op 写、又被若干 op 读 → 这些 op 之间补一条 producer→consumer 边。
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
    # distance 起点 = 自己的 cycles；每次松弛加上后继节点的 cycles。
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
        # 检测到环 → 放弃关键路径，避免误导。
        return 0, []
    end = max(distance, key=distance.get, default=None)
    path: list[str] = []
    while end:
        path.append(end)
        end = previous[end]
    return max(distance.values(), default=0), list(reversed(path))


def summary(case_name: str) -> dict:
    """返回单 case 的概览：节点/边/算子数、关键路径、op/pipe 统计、存储分布。"""
    case = load_case(case_name)
    raw, ops, tensors = case["raw"], case["ops"], case["tensors"]
    path_cycles, path = critical_path(ops, raw["edges"])
    op_counts = Counter(x["op"] for x in raw["ops"])
    pipe_counts = Counter(x["pipe"] for x in raw["ops"])
    storage = Counter()
    storage_bytes = Counter()
    for item in raw["tensors"]:
        # pos：HBM/L2 等存储位置；按位置分桶统计张量数与字节数。
        storage[item["pos"]] += 1
        storage_bytes[item["pos"]] += int(item["size"])
    # 默认焦点：cycle 最高的非 COPY 算子；图空则退化为第一个算子。
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
    """以 focus 为中心做 BFS，截取 radius 跳邻域内的子图。

    参数：
        * focus   —— 焦点 op id；为空时自动取 cycle 最高的非 COPY 算子；
        * radius  —— BFS 跳数上限；
        * limit   —— 最多保留多少节点（防止前端崩溃）；
        * kind    —— "ops"/"tensors"/"all"，过滤节点类型；
        * pipe    —— 仅保留指定 pipe 的算子（focus 自身永远保留）。
    """
    case = load_case(case_name)
    raw, ops, tensors, neighbors = case["raw"], case["ops"], case["tensors"], case["neighbors"]
    candidates = [key for key, item in ops.items() if not pipe or item["pipe"] == pipe]
    if not focus or focus not in ops:
        focus = max(candidates or list(ops), key=lambda key: int(ops[key].get("cycles", 0)))
    start = f"o:{focus}"
    selected = {start}
    frontier = deque([(start, 0)])
    # 标准 BFS；depth 达到 radius 即停。limit 是节点总数上限。
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
        # 仅保留指定类型的节点；focus 自身始终保留，避免空图。
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
    # 边只输出“两个端点都在 selected 内”且 source < target 的边，避免重复。
    edges = [{"source": source, "target": target} for source in selected for target in neighbors.get(source, []) if target in selected and source < target]
    return {"case": case_name, "focus": focus, "nodes": nodes, "edges": edges, "truncated": len(selected) >= limit}


class Handler(BaseHTTPRequestHandler):
    """HTTP 请求处理器：4 个 GET 端点。所有日志被静默。"""

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                # 首页：直接把 HTML 文件原样返回。
                body, content_type = HTML_FILE.read_bytes(), "text/html; charset=utf-8"
            elif parsed.path == "/api/cases":
                # 全部 case 概览列表，供首页侧栏渲染。
                body = json.dumps([summary(path.name) for path in sorted(DATA_DIR.glob("case_*.json"))], ensure_ascii=False).encode()
                content_type = "application/json; charset=utf-8"
            elif parsed.path == "/api/case":
                # 子图切片：focus/radius/limit/kind/pipe 全部从 query 取，并对
                # limit 做了夹紧 (20~800)，避免前端崩溃。
                query = parse_qs(parsed.query)
                case = query.get("case", ["case_001.json"])[0]
                body = json.dumps(graph_slice(case, query.get("focus", [None])[0], int(query.get("radius", [2])[0]), min(800, max(20, int(query.get("limit", [240])[0]))), query.get("kind", ["all"])[0], query.get("pipe", [""])[0]), ensure_ascii=False).encode()
                content_type = "application/json; charset=utf-8"
            elif parsed.path == "/api/ops":
                # 算子列表，用于焦点选择下拉框。
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
            # no-store：每次都拉最新数据，避免开发期间缓存干扰。
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            self.send_error(400, str(exc))

    def log_message(self, format: str, *args: object) -> None:
        # 静默 HTTP 访问日志，否则开发时控制台非常吵。
        return


def main() -> None:
    """CLI 入口：启动 HTTP server，可选自动打开浏览器。"""
    parser = argparse.ArgumentParser(description="Visualize computation graphs in artifacts/data")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Graph visualizer: {url}")
    if not args.no_browser:
        # 延迟 0.4 秒再打开浏览器：给 server 一点启动时间，避免首次加载失败。
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
