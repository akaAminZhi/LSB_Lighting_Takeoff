# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import time
from collections import defaultdict, Counter
from statistics import median
from typing import Dict, List, Tuple, Iterable, Optional, Set

import numpy as np
from sklearn.cluster import DBSCAN
import fitz  # PyMuPDF
from heapq import heappush, heappop

Point = Tuple[float, float]
Edge = Tuple[int, int]
BRANCH_SUBJECT = '(3) #10 IN 3/4" EMT'  # device→JB 固定命名

# ===== 配色（Okabe–Ito，色盲友好） =====
OKABE_ITO = [
    (0.0, 0.447, 0.698),
    (0.902, 0.624, 0.000),
    (0.000, 0.619, 0.451),
    (0.800, 0.475, 0.655),
    (0.941, 0.894, 0.259),
    (0.835, 0.369, 0.000),
    (0.000, 0.000, 0.000),
    (0.600, 0.600, 0.600),
]


def pick_device_colors(labels):
    uniq = sorted(set(labels))
    return {lab: OKABE_ITO[i % len(OKABE_ITO)] for i, lab in enumerate(uniq)}


# ==================== 基础工具 ====================


def manhattan(a: Point, b: Point) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def orthogonal_dogleg(a: Point, b: Point, mode: str = "auto") -> List[Point]:
    if mode == "h":
        bend = (b[0], a[1])
    elif mode == "v":
        bend = (a[0], b[1])
    else:
        bend = (b[0], a[1]) if abs(b[0] - a[0]) <= abs(b[1] - a[1]) else (a[0], b[1])
    pts = [a]
    if bend != a and bend != b:
        pts.append(bend)
    if pts[-1] != b:
        pts.append(b)
    return pts


def l1_median(points: Iterable[Point]) -> Point:
    ps = list(points)
    xs = [p[0] for p in ps]
    ys = [p[1] for p in ps]
    return (median(xs), median(ys))


def snap(p: Point, step: float) -> Point:
    if step and step > 0:
        return (round(p[0] / step) * step, round(p[1] / step) * step)
    return p


# ==================== 线段几何（避墙） ====================


def clamp01(t):
    return 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)


def seg_seg_intersect(p0, p1, q0, q1, *, inclusive=True) -> bool:
    (x1, y1), (x2, y2) = p0, p1
    (x3, y3), (x4, y4) = q0, q1

    def orient(ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    o1 = orient(x1, y1, x2, y2, x3, y3)
    o2 = orient(x1, y1, x2, y2, x4, y4)
    o3 = orient(x3, y3, x4, y4, x1, y1)
    o4 = orient(x3, y3, x4, y4, x2, y2)
    if inclusive:
        if (
            o1 == 0
            and min(x1, x2) - 1e-9 <= x3 <= max(x1, x2) + 1e-9
            and min(y1, y2) - 1e-9 <= y3 <= max(y1, y2) + 1e-9
        ):
            return True
        if (
            o2 == 0
            and min(x1, x2) - 1e-9 <= x4 <= max(x1, x2) + 1e-9
            and min(y1, y2) - 1e-9 <= y4 <= max(y1, y2) + 1e-9
        ):
            return True
        if (
            o3 == 0
            and min(x3, x4) - 1e-9 <= x1 <= max(x3, x4) + 1e-9
            and min(y3, y4) - 1e-9 <= y1 <= max(y3, y4) + 1e-9
        ):
            return True
        if (
            o4 == 0
            and min(x3, x4) - 1e-9 <= x2 <= max(x3, x4) + 1e-9
            and min(y3, y4) - 1e-9 <= y2 <= max(y3, y4) + 1e-9
        ):
            return True
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def _point_to_segment_distance(px, py, ax, ay, bx, by) -> float:
    vx, vy = bx - ax, by - ay
    ux, uy = px - ax, py - ay
    denom = vx * vx + vy * vy
    t = 0.0 if denom == 0 else clamp01((ux * vx + uy) / denom)
    cx, cy = ax + t * vx, ay + t * vy
    dx, dy = px - cx, py - cy
    return math.hypot(dx, dy)


def seg_seg_distance(p0, p1, q0, q1) -> float:
    if seg_seg_intersect(p0, p1, q0, q1):
        return 0.0
    (x1, y1), (x2, y2) = p0, p1
    (x3, y3), (x4, y4) = q0, q1
    return min(
        _point_to_segment_distance(x1, y1, x3, y3, x4, y4),
        _point_to_segment_distance(x2, y2, x3, y3, x4, y4),
        _point_to_segment_distance(x3, y3, x1, y1, x2, y2),
        _point_to_segment_distance(x4, y4, x1, y1, x2, y2),
    )


def point_hits_walls(
    p: Point, walls: List[List[Point]], *, clearance_px: float
) -> bool:
    if not walls:
        return False
    px, py = p
    for poly in walls:
        if len(poly) < 2:
            continue
        for u, v in zip(poly, poly[1:]):
            if (
                _point_to_segment_distance(px, py, u[0], u[1], v[0], v[1])
                < clearance_px - 1e-9
            ):
                return True
    return False


def segment_hits_walls(
    a: Point, b: Point, walls: List[List[Point]], *, clearance_px: float
) -> bool:
    if not walls:
        return False
    for poly in walls:
        if len(poly) < 2:
            continue
        for u, v in zip(poly, poly[1:]):
            if clearance_px <= 1e-6:
                if seg_seg_intersect(a, b, u, v, inclusive=True):
                    return True
            else:
                if seg_seg_distance(a, b, u, v) < clearance_px - 1e-9:
                    return True
    return False


def simplify_orthogonal(poly: List[Point]) -> List[Point]:
    if len(poly) <= 2:
        return poly
    out = [poly[0]]
    for i in range(1, len(poly) - 1):
        x0, y0 = out[-1]
        x1, y1 = poly[i]
        x2, y2 = poly[i + 1]
        if (x0 == x1 == x2) or (y0 == y1 == y2):
            continue
        out.append((x1, y1))
    out.append(poly[-1])
    return out


def nudge_point_off_walls(
    p: Point,
    *,
    walls: List[List[Point]],
    grid_px: float,
    clearance_px: float,
    max_rings: int = 6,
) -> Point:
    if (
        not walls
        or clearance_px <= 0.0
        or not point_hits_walls(p, walls, clearance_px=clearance_px)
    ):
        return p
    x0, y0 = p
    step = grid_px if grid_px > 0 else 50.0
    for r in range(1, max_rings + 1):
        for dx in range(-r, r + 1):
            for dy in (-r, r):
                cand = (
                    round((x0 + dx * step) / step) * step,
                    round((y0 + dy * step) / step) * step,
                )
                if not point_hits_walls(cand, walls, clearance_px=clearance_px):
                    return cand
        for dy in range(-r + 1, r):
            for dx in (-r, r):
                cand = (
                    round((x0 + dx * step) / step) * step,
                    round((y0 + dy * step) / step) * step,
                )
                if not point_hits_walls(cand, walls, clearance_px=clearance_px):
                    return cand
    return p


# ==================== A*（避墙，带硬停） ====================


def a_star_rectilinear(
    start: Point,
    goal: Point,
    *,
    grid_px: float,
    walls: List[List[Point]],
    clearance_px: float,
    bbox_margin_cells: int = 6,
    node_limit: int = 20000,
    deadline_s: float = 0.25,
) -> Optional[List[Point]]:
    if grid_px <= 0:
        grid_px = 50.0

    def snap_to_grid(p):
        return (round(p[0] / grid_px) * grid_px, round(p[1] / grid_px) * grid_px)

    s = snap_to_grid(start)
    g = snap_to_grid(goal)
    minx = min(s[0], g[0]) - bbox_margin_cells * grid_px
    maxx = max(s[0], g[0]) + bbox_margin_cells * grid_px
    miny = min(s[1], g[1]) - bbox_margin_cells * grid_px
    maxy = max(s[1], g[1]) + bbox_margin_cells * grid_px

    def neighbors(p):
        x, y = p
        for nx, ny in [
            (x + grid_px, y),
            (x - grid_px, y),
            (x, y + grid_px),
            (x, y - grid_px),
        ]:
            if nx < minx or nx > maxx or ny < miny or ny > maxy:
                continue
            if not segment_hits_walls(
                (x, y), (nx, ny), walls, clearance_px=clearance_px
            ):
                yield (nx, ny)

    def h(p):
        return abs(p[0] - g[0]) + abs(p[1] - g[1])

    t0 = time.time()
    openq = []
    heappush(openq, (h(s), 0.0, s))
    came = {}
    gscore = {s: 0.0}
    visited = set()
    expanded = 0
    while openq:
        if expanded > node_limit or (time.time() - t0) > deadline_s:
            return None
        _, gs, cur = heappop(openq)
        if cur in visited:
            continue
        visited.add(cur)
        expanded += 1
        if cur == g:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return simplify_orthogonal(path)
        for nb in neighbors(cur):
            tentative = gs + grid_px
            if tentative < gscore.get(nb, float("inf")):
                gscore[nb] = tentative
                came[nb] = cur
                heappush(openq, (tentative + h(nb), tentative, nb))
    return None


def route_rect_with_walls(
    a: Point,
    b: Point,
    *,
    walls: List[List[Point]],
    grid_px: float,
    clearance_px: float,
    bbox_margin_cells: int = 8,
    max_expand_rounds: int = 1,
    max_refine_rounds: int = 1,
    node_limit: int = 15000,
    deadline_s: float = 0.20,
) -> List[Point]:
    cand = orthogonal_dogleg(a, b, mode="auto")
    ok = True
    for u, v in zip(cand, cand[1:]):
        if segment_hits_walls(u, v, walls, clearance_px=clearance_px):
            ok = False
            break
    if ok:
        return cand
    for expand_i in range(max_expand_rounds + 1):
        cur_margin = bbox_margin_cells * (2**expand_i)
        cur_grid = grid_px
        for _ in range(max_refine_rounds + 1):
            path = a_star_rectilinear(
                a,
                b,
                grid_px=cur_grid,
                walls=walls,
                clearance_px=clearance_px,
                bbox_margin_cells=cur_margin,
                node_limit=node_limit,
                deadline_s=deadline_s,
            )
            if path:
                return path
            cur_grid = max(cur_grid / 2.0, 5.0)
    return cand


# ==================== PDF 提取设备 ====================


def get_word_centers(
    page: fitz.Page, target_text: str, case_sensitive: bool = True
) -> List[Point]:
    words = page.get_text("words")
    cs = []
    if not case_sensitive:
        target_text = target_text.lower()
    for x0, y0, x1, y1, w, *_ in words:
        token = w if case_sensitive else w.lower()
        if token == target_text:
            cs.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
    return cs


def build_dets_from_pdf_text_multi(
    page: fitz.Page, device_labels: List[str]
) -> List[Dict]:
    dets_all = []
    for lab in device_labels:
        for x, y in get_word_centers(page, lab, case_sensitive=True):
            dets_all.append(
                {
                    "x1": x - 2,
                    "y1": y - 2,
                    "x2": x + 2,
                    "y2": y + 2,
                    "label": lab,
                    "confidence": 1.0,
                }
            )
    return dets_all


# ==================== 家族（≤3，含锁定） ====================


def build_disjoint_families(
    dets: List[Dict],
    keep_labels: Iterable[str],
    *,
    neighbor_eps_px: float = 400.0,
    min_conf: float = 0.30,
    priority_families: Optional[List[Iterable[str]]] = None,
    max_family_size: int = 3,
) -> Tuple[Dict[str, int], List[Set[str]]]:
    """
    priority_families 作为“锁定家族”（allowed set）。跨允许集或 >max_family_size 的合并将被拒绝。
    """
    keep_labels = set(keep_labels)

    # 共现计数
    pts, labs = [], []
    for d in dets:
        if d.get("confidence", 1.0) < min_conf:
            continue
        lab = d.get("label")
        if lab not in keep_labels:
            continue
        cx = 0.5 * (d["x1"] + d["x2"])
        cy = 0.5 * (d["y1"] + d["y2"])
        pts.append((cx, cy))
        labs.append(lab)

    n = len(pts)
    co = Counter()
    for i in range(n):
        xi, yi = pts[i]
        li = labs[i]
        for j in range(i + 1, n):
            if abs(xi - pts[j][0]) + abs(yi - pts[j][1]) <= neighbor_eps_px:
                lj = labs[j]
                if li == lj:
                    continue
                a, b = sorted((li, lj))
                co[(a, b)] += 1

    labels = sorted({lab for lab in labs})
    if not labels:
        return {}, []

    idx_of_label = {lab: i for i, lab in enumerate(labels)}

    parent = list(range(len(labels)))
    rank = [0] * len(labels)
    comp_labels: List[Set[str]] = [{lab} for lab in labels]
    comp_allowed: List[Optional[Set[str]]] = [None] * len(labels)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def can_union_roots(rx: int, ry: int) -> bool:
        U = comp_labels[rx] | comp_labels[ry]
        if len(U) > max_family_size:
            return False
        A = comp_allowed[rx]
        B = comp_allowed[ry]
        if A is not None and not U.issubset(A):
            return False
        if B is not None and not U.issubset(B):
            return False
        return True

    def union_roots(rx: int, ry: int):
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        comp_labels[rx] |= comp_labels[ry]
        A = comp_allowed[rx]
        B = comp_allowed[ry]
        if A is None and B is None:
            comp_allowed[rx] = None
        elif A is None and B is not None:
            comp_allowed[rx] = set(B)
        elif A is not None and B is None:
            comp_allowed[rx] = set(A)
        else:
            comp_allowed[rx] = set(A) & set(B)
        return rx

    # 锁定：把 priority_families 变成 allowed set，并预先在集合内部合并
    if priority_families:
        locked = []
        for fam in priority_families:
            s = sorted(set(fam) & set(labels))
            if s:
                locked.append(set(s))
        for allow in locked:
            for lab in allow:
                r = idx_of_label[lab]
                comp_allowed[r] = set(allow)
        for allow in locked:
            labs_sorted = sorted(allow)
            base = idx_of_label[labs_sorted[0]]
            for lab in labs_sorted[1:]:
                rx, ry = find(base), find(idx_of_label[lab])
                if rx != ry and can_union_roots(rx, ry):
                    base = union_roots(rx, ry)

    # 按共现由大到小尝试合并
    for (a, b), _w in co.most_common():
        if a not in idx_of_label or b not in idx_of_label:
            continue
        rx, ry = find(idx_of_label[a]), find(idx_of_label[b])
        if rx != ry and can_union_roots(rx, ry):
            union_roots(rx, ry)

    # 导出 families
    rep_map: Dict[int, int] = {}
    families: List[Set[str]] = []
    for lab, i in idx_of_label.items():
        r = find(i)
        if r not in rep_map:
            rep_map[r] = len(families)
            families.append(set())
        families[rep_map[r]].add(lab)

    family_id_of_label = {lab: rep_map[find(idx_of_label[lab])] for lab in labels}
    return family_id_of_label, families


# ==================== JB 放置（含避墙） ====================


def auto_place_junction_boxes(
    dets: List[Dict],
    *,
    capacity: int = 8,
    eps_px: float = 400,
    grid_px: float = 50,
    merge_eps: float = 50,
    min_conf: float = 0.30,
    keep_labels: Optional[Iterable[str]] = None,
    return_coords: bool = False,
    priority_families: Optional[List[Iterable[str]]] = None,
    walls: Optional[List[List[Point]]] = None,
    clearance_px: float = 0.0,
    max_dev_jb_ft: Optional[float] = 6.0,  # 新增：设备→JB 最大允许长度（单位：ft）
):
    """
    自动放置 JB，增加约束：
      - 同一个 JB 下所有设备，到 JB 的 L1 距离不超过 max_dev_jb_ft（默认 6 ft）。
      - JB 合并时会保证存在一个“公共 L1 半径菱形交集”，并把 JB 放在该交集区域内。

    其它行为与原版保持一致：
      - 同一 label 近距离的多个框先通过 DBSCAN 合并为一个设备点；
      - JB 合并仍然受 capacity、family（≤3 种类型、同 family） 等限制。
    """
    if keep_labels is None:
        all_labels = {d["label"] for d in dets if d.get("label") is not None}
        keep_labels = {l for l in all_labels if l != "PANEL"}
    keep_labels = set(keep_labels)

    # 家族划分（同 family 且 ≤3 种类型）
    family_id_of_label, families = build_disjoint_families(
        dets,
        keep_labels,
        neighbor_eps_px=eps_px,
        min_conf=min_conf,
        priority_families=priority_families,
        max_family_size=3,
    )

    # ===== 设备点抽取 =====
    raw_pts: List[Point] = []
    raw_idx: List[int] = []  # 对应 dets 的下标
    raw_labs: List[str] = []
    det_to_raw: Dict[int, int] = {}  # det index -> raw_pts index

    for i, d in enumerate(dets):
        if d.get("confidence", 1.0) < min_conf:
            continue
        lab = d.get("label")
        if lab not in keep_labels:
            continue
        cx = 0.5 * (d["x1"] + d["x2"])
        cy = 0.5 * (d["y1"] + d["y2"])
        raw_pts.append((cx, cy))
        det_to_raw[i] = len(raw_pts) - 1
        raw_idx.append(i)
        raw_labs.append(lab)

    if not raw_pts:
        raise ValueError("No valid devices after filtering")

    # ===== 同标签、近距离设备用 DBSCAN 合并为“设备点” =====
    dev_points: List[Point] = []
    dev_label_sets: List[Set[str]] = []
    dev_counts: List[int] = []
    dev_point_to_global_idxs: List[List[int]] = []  # 每个设备点对应哪些 det 索引

    by_label: Dict[str, List[int]] = defaultdict(list)
    for k, lab in enumerate(raw_labs):
        by_label[lab].append(k)

    for lab, idxs in by_label.items():
        pts_lab = np.asarray([raw_pts[k] for k in idxs], float)
        if len(pts_lab) == 0:
            continue
        labels_lab = DBSCAN(eps=merge_eps, min_samples=1).fit_predict(pts_lab)
        for val in np.unique(labels_lab):
            sub = np.where(labels_lab == val)[0]
            members = [idxs[s] for s in sub]  # 这些是 raw_pts 的索引
            p = pts_lab[sub].mean(axis=0)
            dev_points.append((float(p[0]), float(p[1])))
            dev_label_sets.append({lab})
            dev_counts.append(len(members))
            # 记录回到 dets 的下标
            dev_point_to_global_idxs.append([raw_idx[m] for m in members])

    dev_points_np = np.asarray(dev_points, float)
    INF = 1e18

    # ===== 把“6 ft 限制”换算到 PDF 坐标（pt），并为每个初始 JB 建 L1 菱形交集 =====
    # L1 距离约束：|x - xi| + |y - yi| <= R_pts
    # 用 S=x+y, D=x-y 转成区间交集，便于合并时快速判断是否还存在公共区域。
    if (
        max_dev_jb_ft is not None
        and max_dev_jb_ft > 0
        and "PTS_PER_FT" in globals()
        and PTS_PER_FT > 0
        and len(dev_points) > 0
    ):
        max_dev_jb_pts: Optional[float] = max_dev_jb_ft * PTS_PER_FT
    else:
        max_dev_jb_pts = None

    if max_dev_jb_pts is not None:
        raw_pts_np = np.asarray(raw_pts, float)
        raw_S = raw_pts_np[:, 0] + raw_pts_np[:, 1]
        raw_D = raw_pts_np[:, 0] - raw_pts_np[:, 1]

        # 每个“初始 JB”（一个设备点）对应一个 L1 菱形交集区域：
        #   JB_diamonds[i] = (S_lo, S_hi, D_lo, D_hi)
        JB_diamonds: List[Optional[Tuple[float, float, float, float]]] = []
        for g_indices in dev_point_to_global_idxs:
            S_lo, S_hi = -INF, INF
            D_lo, D_hi = -INF, INF
            for g in g_indices:
                r = det_to_raw[g]
                S = raw_S[r]
                D = raw_D[r]
                S_lo = max(S_lo, S - max_dev_jb_pts)
                S_hi = min(S_hi, S + max_dev_jb_pts)
                D_lo = max(D_lo, D - max_dev_jb_pts)
                D_hi = min(D_hi, D + max_dev_jb_pts)
            if S_lo > S_hi or D_lo > D_hi:
                # 理论上不会发生；兜底成一个“退化区间”
                JB_diamonds.append((S_lo, S_lo, D_lo, D_lo))
            else:
                JB_diamonds.append((S_lo, S_hi, D_lo, D_hi))
    else:
        JB_diamonds = [None] * len(dev_points)

    # ===== 初始化 JB：一开始每个设备点一个 JB =====
    JB_centers: List[Point] = []
    for p in dev_points_np:
        snapped = snap((float(p[0]), float(p[1])), grid_px)
        JB_centers.append(
            nudge_point_off_walls(
                snapped, walls=walls or [], grid_px=grid_px, clearance_px=clearance_px
            )
        )
    JB_label_sets: List[Set[str]] = [set(s) for s in dev_label_sets]
    JB_counts: List[int] = list(dev_counts)
    JB_members: List[List[int]] = [[i] for i in range(len(dev_points))]
    active: List[bool] = [True] * len(JB_centers)

    # ===== family 判定（沿用你之前的逻辑） =====
    def same_family(L: Set[str]) -> bool:
        if not L:
            return True
        fids = {family_id_of_label.get(t) for t in L if t in family_id_of_label}
        fids.discard(None)
        if len(fids) != 1:
            return False
        fid = next(iter(fids))
        return L.issubset(families[fid])

    # ===== 是否允许合并两个 JB =====
    def can_merge(i: int, j: int) -> bool:
        if i == j or (not active[i]) or (not active[j]):
            return False
        # 距离、容量、family 类型数量等原本条件
        if manhattan(JB_centers[i], JB_centers[j]) > eps_px:
            return False
        if JB_counts[i] + JB_counts[j] > capacity:
            return False
        L = JB_label_sets[i] | JB_label_sets[j]
        if len(L) > 3:
            return False
        if not same_family(L):
            return False

        # === 新增：6 ft 约束对应的 L1 菱形交集判断 ===
        if max_dev_jb_pts is not None:
            di = JB_diamonds[i]
            dj = JB_diamonds[j]
            if di is None or dj is None:
                return False
            S_lo = max(di[0], dj[0])
            S_hi = min(di[1], dj[1])
            D_lo = max(di[2], dj[2])
            D_hi = min(di[3], dj[3])
            if S_lo > S_hi or D_lo > D_hi:
                # 说明不存在一个点，能让“合并后的所有设备”到 JB 均 ≤ max_dev_jb_ft
                return False

        return True

    def merge_score(i: int, j: int) -> Tuple[int, float, float]:
        Li, Lj = JB_label_sets[i], JB_label_sets[j]
        inter = len(Li & Lj)  # 交集越大越好
        new_added = len((Li | Lj)) - max(len(Li), len(Lj))  # 新增类型越少越好
        dist = manhattan(JB_centers[i], JB_centers[j])  # 距离越近越好
        return (inter, -new_added, -dist)

    # ===== 贪心合并 JB（加入 6 ft 约束） =====
    while True:
        best = None
        best_pair = (-1, -1)
        n = len(JB_centers)
        for i in range(n):
            if not active[i]:
                continue
            for j in range(i + 1, n):
                if not active[j]:
                    continue
                if not can_merge(i, j):
                    continue
                s = merge_score(i, j)
                if (best is None) or (s > best):
                    best = s
                    best_pair = (i, j)

        if best is None:
            break

        i, j = best_pair

        # 合并成员 / 数量 / 类型集合
        new_members = JB_members[i] + JB_members[j]
        JB_members[i] = new_members
        JB_counts[i] += JB_counts[j]
        JB_label_sets[i] |= JB_label_sets[j]

        # === 新：计算合并后 JB 的 L1 菱形交集，并用交集区域的“中心点”作为 JB 新位置 ===
        if max_dev_jb_pts is not None:
            di = JB_diamonds[i]
            dj = JB_diamonds[j]
            if di is None or dj is None:
                # 保险起见，从头根据所有成员重新算一个 diamond
                S_lo, S_hi = -INF, INF
                D_lo, D_hi = -INF, INF
                for dp_idx in new_members:
                    for g in dev_point_to_global_idxs[dp_idx]:
                        r = det_to_raw[g]
                        # 这里 raw_S / raw_D 在 max_dev_jb_pts is not None 的分支中已定义
                        S = raw_S[r]
                        D = raw_D[r]
                        S_lo = max(S_lo, S - max_dev_jb_pts)
                        S_hi = min(S_hi, S + max_dev_jb_pts)
                        D_lo = max(D_lo, D - max_dev_jb_pts)
                        D_hi = min(D_hi, D + max_dev_jb_pts)
            else:
                # 直接用两个 diamond 的交集
                S_lo = max(di[0], dj[0])
                S_hi = min(di[1], dj[1])
                D_lo = max(di[2], dj[2])
                D_hi = min(di[3], dj[3])

            JB_diamonds[i] = (S_lo, S_hi, D_lo, D_hi)

            # 在交集区域取一个中点：S_c, D_c → (x_c, y_c)
            S_c = 0.5 * (S_lo + S_hi)
            D_c = 0.5 * (D_lo + D_hi)
            new_center = (
                0.5 * (S_c + D_c),
                0.5 * (S_c - D_c),
            )
        else:
            # 不启用 6 ft 约束则保持之前“平均位置”的逻辑
            new_center = tuple(
                np.asarray([dev_points_np[k] for k in new_members], float).mean(axis=0)
            )

        snapped = snap(new_center, grid_px)
        JB_centers[i] = nudge_point_off_walls(
            snapped, walls=walls or [], grid_px=grid_px, clearance_px=clearance_px
        )
        active[j] = False

    # ===== 导出结果 =====
    jb_xy: List[Point] = []
    jb_label_sets_out: List[Set[str]] = []
    jb_members_out: List[List[int]] = []
    for old_i, on in enumerate(active):
        if not on:
            continue
        jb_xy.append(JB_centers[old_i])
        jb_label_sets_out.append(JB_label_sets[old_i])
        jb_members_out.append(JB_members[old_i])

    # dev index → JB index 映射
    dev2jb: Dict[int, int] = {}
    groups: Dict[int, List[int]] = defaultdict(list)
    for new_jb_idx, members in enumerate(jb_members_out):
        for dp in members:
            for g in dev_point_to_global_idxs[dp]:
                dev2jb[g] = new_jb_idx
                groups[new_jb_idx].append(g)

    if return_coords:
        dev_coords = [tuple(p) for p in raw_pts]
        return (
            jb_xy,
            dev2jb,
            dict(groups),
            dev_coords,
            jb_label_sets_out,
            family_id_of_label,
            families,
        )
    return jb_xy, dev2jb, dict(groups), jb_label_sets_out, family_id_of_label, families


# ==================== 注释绘制（含测量模板缓存） ====================

TEMPLATE_CACHE: Dict[int, Dict[str, str]] = {}


def _xref_get(doc: fitz.Document, xref: int, key: str):
    try:
        t, v = doc.xref_get_key(xref, key)
        return (t or "null"), v
    except Exception:
        return "null", None


def init_polyline_template_cache(
    doc: fitz.Document,
    template_subject: str = "template",
) -> None:
    did = id(doc)
    if did in TEMPLATE_CACHE and TEMPLATE_CACHE[did].get("subject") == template_subject:
        return
    tpl_page_idx = None
    tpl_annot = None
    for pg_idx in range(len(doc)):
        page = doc[pg_idx]
        for a in page.annots() or []:
            if a.type[1] != "PolyLine":
                continue
            subj = (a.info or {}).get("subject") or ""
            if subj.strip().lower() == template_subject.lower():
                tpl_page_idx = pg_idx
                tpl_annot = a
                break
        if tpl_annot:
            break
    if tpl_annot is None:
        raise RuntimeError(f"未找到 PolyLine 模板（subject='{template_subject}'）")

    it_type, it_val = _xref_get(doc, tpl_annot.xref, "IT")
    it_value = it_val if it_val else "/PolyLineDimension"

    m_type, m_val = _xref_get(doc, tpl_annot.xref, "Measure")
    if not m_val:
        measure_ref = None
    else:
        if m_type == "xref":
            measure_ref = m_val
        elif m_type == "dict":
            new_m = doc.get_new_xref()
            doc.update_object(new_m, m_val)
            measure_ref = f"{new_m} 0 R"
        else:
            measure_ref = None

    TEMPLATE_CACHE[did] = {
        "subject": template_subject,
        "it_value": it_value,
        "measure_ref": measure_ref,
        "page_index": str(tpl_page_idx),
        "annot_xref": str(tpl_annot.xref),
    }


def draw_circle_annot(
    page: fitz.Page,
    center: Point,
    r: float,
    stroke_color=(1, 0, 0),
    fill_color=None,
    width: float = 0.8,
    title: str = "",
    contents: str = "",
):
    x, y = center
    rect = fitz.Rect(x - r, y - r, x + r, y + r)
    annot = page.add_circle_annot(rect)
    annot.set_blendmode("Multiply")
    annot.set_colors(stroke=stroke_color, fill=fill_color)
    annot.set_border(width=width)
    annot.set_info(subject=title)
    if title:
        annot.set_info(title=title)
    if contents:
        annot.set_info(content=contents)
    annot.update()
    return annot


def draw_polyline_annot(
    page: fitz.Page,
    pts: List[Point],
    width: Optional[float] = 0.9,
    stroke_color: Optional[Tuple[float, float, float]] = (1, 0, 0),
    title: str = "WIRE",
    contents: str = "",
    template_subject: str = "template",
    copy_template_appearance: bool = True,
) -> Optional[fitz.Annot]:
    if len(pts) < 2:
        return None
    doc = page.parent
    did = id(doc)
    cache = TEMPLATE_CACHE.get(did)
    if not cache or cache.get("subject") != template_subject:
        init_polyline_template_cache(doc, template_subject)
        cache = TEMPLATE_CACHE.get(did)

    plist = [fitz.Point(p[0], p[1]) for p in pts]
    annot = page.add_polyline_annot(plist)

    it_value = cache.get("it_value")
    if it_value:
        try:
            doc.xref_set_key(annot.xref, "IT", it_value)
        except Exception:
            pass
    measure_ref = cache.get("measure_ref")
    if measure_ref:
        try:
            doc.xref_set_key(annot.xref, "Measure", measure_ref)
        except Exception:
            pass

    if copy_template_appearance:
        try:
            tpl_page_idx = int(cache.get("page_index"))
            tpl_xref = int(cache.get("annot_xref"))
            tpl_page = doc[tpl_page_idx]
            tpl_annot = None
            for a in tpl_page.annots() or []:
                if a.xref == tpl_xref:
                    tpl_annot = a
                    break
            if tpl_annot:
                try:
                    colors = (tpl_annot.colors or {}).copy()
                    if colors:
                        annot.set_colors(
                            stroke=colors.get("stroke"), fill=colors.get("fill")
                        )
                except Exception:
                    pass
                try:
                    border = (tpl_annot.border or {}).copy()
                    if border:
                        annot.set_border(
                            width=border.get("width"), dashes=border.get("dashes")
                        )
                except Exception:
                    pass
                try:
                    flags = getattr(tpl_annot, "flags", None)
                    if flags is not None:
                        annot.set_flags(flags)
                except Exception:
                    pass
                try:
                    opacity = getattr(tpl_annot, "opacity", None)
                    if opacity is not None:
                        annot.set_opacity(opacity)
                except Exception:
                    pass
                try:
                    le = getattr(tpl_annot, "line_ends", None)
                    if le:
                        if isinstance(le, dict):
                            annot.set_line_ends(le.get("start"), le.get("end"))
                        elif isinstance(le, (list, tuple)) and len(le) >= 2:
                            annot.set_line_ends(le[0], le[1])
                except Exception:
                    pass
        except Exception:
            pass

    try:
        if stroke_color is not None:
            annot.set_colors(stroke=stroke_color)
    except Exception:
        pass
    try:
        if width is not None:
            annot.set_border(width=width)
    except Exception:
        pass

    try:
        annot.set_info(subject=title)
        if title:
            annot.set_info(title=title)
        if contents:
            annot.set_info(content=contents)
    except Exception:
        pass
    try:
        annot.set_blendmode("Multiply")
    except Exception:
        pass
    return annot


# ==================== 并排走线（lane 分配 + 平移，可选重叠） ====================


def _norm_seg(a: Point, b: Point) -> Tuple[Point, Point]:
    return (a, b) if a <= b else (b, a)


def _poly_to_base_segs(poly: List[Point]) -> List[Tuple[Point, Point]]:
    segs = []
    for u, v in zip(poly, poly[1:]):
        if u != v:
            segs.append(_norm_seg(u, v))
    return segs


def _lane_order():
    yield 0
    k = 1
    while True:
        yield k
        yield -k
        k += 1


def offset_poly_axiswise(
    poly: List[Point], lane_idx: int, lane_gap: float
) -> List[Point]:
    """
    轴向友好平移：
      - 水平段整体仅在 y 方向平移
      - 垂直段整体仅在 x 方向平移
      - 拐角点取相邻点平移结果的“过渡中点”以避免缺口
    """
    if lane_idx == 0 or abs(lane_gap) < 1e-6 or len(poly) < 2:
        return poly[:]

    def seg_dir(a: Point, b: Point) -> Tuple[int, int]:
        if a[0] == b[0]:
            return (0, 1)  # vertical
        elif a[1] == b[1]:
            return (1, 0)  # horizontal
        else:
            return (1, 0)  # 容错，按水平处理

    shifted = []
    n = len(poly)
    for i in range(n):
        if i == 0:
            d = seg_dir(poly[i], poly[i + 1])
        elif i == n - 1:
            d = seg_dir(poly[i - 1], poly[i])
        else:
            d_prev = seg_dir(poly[i - 1], poly[i])
            d_next = seg_dir(poly[i], poly[i + 1])
            if d_prev == d_next:
                d = d_prev
            else:
                d = (max(d_prev[0], d_next[0]), max(d_prev[1], d_next[1]))
        dx = lane_idx * lane_gap * (1 if d[0] == 1 else 0)
        dy = lane_idx * lane_gap * (1 if d[1] == 1 else 0)
        shifted.append((poly[i][0] + dx, poly[i][1] + dy))

    out = [shifted[0]]
    for i in range(1, n - 1):
        (x0, y0), (x1, y1), (x2, y2) = shifted[i - 1], shifted[i], shifted[i + 1]
        if (x0 == x1 == x2) or (y0 == y1 == y2):
            out.append((x1, y1))
        else:
            out.append(((x0 + x2) / 2.0, (y0 + y2) / 2.0))
    out.append(shifted[-1])
    return out


def offset_poly_diag(poly: List[Point], lane_idx: int, lane_gap: float) -> List[Point]:
    """对角同量平移：x、y 同时加 lane_idx*lane_gap"""
    if lane_idx == 0 or abs(lane_gap) < 1e-6:
        return poly[:]
    dx = lane_idx * lane_gap
    dy = lane_idx * lane_gap
    return [(x + dx, y + dy) for (x, y) in poly]


def snap_poly_to_grid(poly: List[Point], grid_px: float) -> List[Point]:
    return [snap(p, grid_px) for p in poly]


def _explode_to_unit_segs(
    poly: List[Point], grid_px: float
) -> List[Tuple[Point, Point]]:
    """把折线按网格粒度拆成单元段，便于‘段级别’占道"""
    if len(poly) < 2:
        return []
    segs = []
    for (x0, y0), (x1, y1) in zip(poly, poly[1:]):
        if x0 == x1:
            y_start, y_end = (y0, y1) if y0 <= y1 else (y1, y0)
            y = y_start
            while y < y_end - 1e-9:
                a = (x0, y)
                b = (x0, min(y + grid_px, y_end))
                if a != b:
                    segs.append(_norm_seg(a, b))
                y += grid_px
        elif y0 == y1:
            x_start, x_end = (x0, x1) if x0 <= x1 else (x1, x0)
            x = x_start
            while x < x_end - 1e-9:
                a = (x, y0)
                b = (min(x + grid_px, x_end), y0)
                if a != b:
                    segs.append(_norm_seg(a, b))
                x += grid_px
        else:
            segs.append(_norm_seg((x0, y0), (x1, y1)))
    return segs


def choose_lane_for_poly(
    poly: List[Point],
    lane_map: Dict[Tuple[Point, Point], Set[int]],
    grid_px: float,
    allow_overlap: bool,
    update_map: bool = True,
) -> int:
    """
    allow_overlap=True 时直接返回 0（不占 lane，不平移）；
    否则：snap + 单元段切分，避免与已有单元段重叠，分配最近未使用的 lane。

    update_map=False 时，只“参考” lane_map 中已有占用情况，但不把本次分配写回 lane_map。
    """
    if allow_overlap:
        return 0

    poly_g = snap_poly_to_grid(poly, grid_px)
    cell_segs = _explode_to_unit_segs(poly_g, grid_px)

    used: Set[int] = set()
    for seg in cell_segs:
        used |= lane_map[seg]

    for cand in _lane_order():
        if cand not in used:
            if update_map:
                for seg in cell_segs:
                    lane_map[seg].add(cand)
            return cand
    return 0


def build_trunk_chains_from_tree(
    tree_adj: Dict[Point, Set[Point]],
    edge_circuits: Dict[Tuple[Point, Point], Set[str]],
    *,
    root: Optional[Point] = None,
) -> List[Tuple[List[Point], Set[str]]]:
    """
    在 trunk 树上，把“回路集合完全相同”的连续边合并成一条 polyline。
    返回：
      [
        ([node0, node1, node2, ...], circuits_set),
        ...
      ]

    - root 用来表示 Panel 的节点；在 root 处不跨过继续合并，
      防止把 Panel 两侧的分支合成一条奇怪的“JBRx→JB Ry”主干。
    """

    def _norm_seg(a: Point, b: Point) -> Tuple[Point, Point]:
        return (a, b) if a <= b else (b, a)

    visited_edges: Set[Tuple[Point, Point]] = set()
    chains: List[Tuple[List[Point], Set[str]]] = []

    for u, nbrs in tree_adj.items():
        for v in nbrs:
            key = _norm_seg(u, v)
            if key in visited_edges:
                continue

            circuits = edge_circuits.get(key)
            # 没有任何回路经过这条边，就标记为 visited 但不画
            if not circuits:
                visited_edges.add(key)
                continue

            # 当前链条以 u-v 为起点，后面向两端延伸
            chain = [u, v]
            visited_edges.add(key)

            def extend(front: bool) -> None:
                nonlocal chain
                if front:
                    cur, prev = chain[0], chain[1]
                else:
                    cur, prev = chain[-1], chain[-2]

                while True:
                    # 到 Panel 节点就停，不跨 Panel 合并
                    if root is not None and cur == root:
                        break

                    candidates = []
                    for w in tree_adj.get(cur, ()):
                        if w == prev:
                            continue
                        k2 = _norm_seg(cur, w)
                        if k2 in visited_edges:
                            continue
                        if edge_circuits.get(k2) == circuits:
                            candidates.append(w)

                    # 只有唯一候选才延伸；否则认为到达“分叉点/终点”
                    if len(candidates) != 1:
                        break

                    w = candidates[0]
                    visited_edges.add(_norm_seg(cur, w))
                    if front:
                        chain.insert(0, w)
                    else:
                        chain.append(w)
                    prev, cur = cur, w

            # 两端各自尽量延伸
            extend(front=True)
            extend(front=False)

            chains.append((chain, circuits))

    return chains


# ==================== 长度格式化 ====================

SCALE_IN_PER_FT = 1.0 / 8.0
PTS_PER_INCH = 72.0
PTS_PER_FT = PTS_PER_INCH * SCALE_IN_PER_FT  # 9 pt/ft


def rect_path_length(pts):
    if not pts or len(pts) < 2:
        return 0.0
    total = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        total += abs(x1 - x0) + abs(y1 - y0)
    return total


def format_length_ft_in(
    L_pts: float, *, pts_per_ft: float = PTS_PER_FT, inch_precision: int = 0
) -> str:
    if pts_per_ft <= 0:
        return f"{L_pts:.1f} pt"
    feet_float = L_pts / pts_per_ft
    whole_ft = int(feet_float)
    inches = (feet_float - whole_ft) * 12.0
    factor = 10**inch_precision
    inches = round(inches * factor) / factor
    if inches >= 12.0:
        whole_ft += 1
        inches -= 12.0
    inches_str = (
        f'{int(inches)}"' if inch_precision == 0 else f'{inches:.{inch_precision}f}"'
    )
    return f"{whole_ft}'-{inches_str}"


# ==================== 组件划分（同家族且 ≤3 种） ====================


def components_by_family(
    jb_xy: List[Point],
    jb_label_sets: List[Set[str]],
    *,
    max_types: int,
    family_id_of_label: Dict[str, int],
    families: List[Set[str]],
) -> List[List[int]]:
    m = len(jb_xy)
    if m == 0:
        return []
    parent = list(range(m))
    rank = [0] * m
    comp_labels = [set(s) for s in jb_label_sets]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def same_family(L: Set[str]) -> Optional[int]:
        if not L:
            return None
        fids = {family_id_of_label.get(t) for t in L if t in family_id_of_label}
        fids.discard(None)
        if len(fids) != 1:
            return None
        fid = next(iter(fids))
        return fid if L.issubset(families[fid]) else None

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return rx
        U = comp_labels[rx] | comp_labels[ry]
        if len(U) > max_types or same_family(U) is None:
            return rx
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        comp_labels[rx] = U
        return rx

    pairs = []
    for i in range(m):
        for j in range(i + 1, m):
            pairs.append((manhattan(jb_xy[i], jb_xy[j]), i, j))
    pairs.sort(key=lambda t: t[0])
    for _, i, j in pairs:
        union(i, j)
    buckets = defaultdict(list)
    for v in range(m):
        buckets[find(v)].append(v)
    return list(buckets.values())


# ==================== 史坦纳主干（严格 ≤3 种） ====================


def _sample_nodes_on_polyline(poly: List[Point], grid_px: float) -> List[Point]:
    if len(poly) < 2:
        return poly[:]
    step = grid_px if grid_px > 0 else 50.0
    out = [poly[0]]
    for (x0, y0), (x1, y1) in zip(poly, poly[1:]):
        if x0 == x1:
            y = min(y0, y1)
            y_end = max(y0, y1)
            k = 1
            while y + k * step < y_end - 1e-6:
                out.append((x0, round((y + k * step) / step) * step))
                k += 1
        elif y0 == y1:
            x = min(x0, x1)
            x_end = max(x0, x1)
            k = 1
            while x + k * step < x_end - 1e-6:
                out.append((round((x + k * step) / step) * step, y0))
                k += 1
        out.append((x1, y1))
    uniq = []
    seen = set()
    for p in out:
        if p in seen:
            continue
        uniq.append(p)
        seen.add(p)
    return uniq


def mainbus_subject_for_type_count(k: int) -> str:
    k_capped = max(1, min(3, int(k)))
    wires = 2 * k_capped + 1
    return f'({wires}) #10 IN 3/4" EMT'


def normalize_device_label(lab: str) -> str:
    """
    把设备 label 归一化成“回路 key”：
      - 现在简单返回 strip 之后的原值，比如 "10" -> "10"
      - 如果以后有 "10A" / "10B" 也当成同一路，可以在这里把尾巴去掉：
          比如用正则提取开头数字部分。
    """
    if lab is None:
        return ""
    return str(lab).strip()


def mainbus_width_for_type_count(k: int) -> float:
    """
    根据类型数量 k（1~3），返回主干线的绘制宽度。
    和 mainbus_subject_for_type_count 保持一致：
      k=1 -> 3 条线
      k=2 -> 5 条线
      k=3 -> 7 条线
    这里给个简单线性规则：
      3 条线 -> 3.0 pt
      5 条线 -> 4.0 pt
      7 条线 -> 5.0 pt
    你可以按喜好调整公式。
    """
    k_capped = max(1, min(3, int(k)))
    wires = 2 * k_capped + 1  # 3 / 5 / 7

    # 简单线性：从 3 条线开始，每多 2 条线宽度 +1pt
    base_width = 2.0  # 3 条线的宽度
    step_per_2w = 2.0  # 每多 2 条线加多少宽度

    extra_pairs = (wires - 3) / 2.0  # 3->0, 5->1, 7->2
    width = base_width + extra_pairs * step_per_2w
    return width


def build_steiner_trunk_paths(
    panel_coord: Point,
    jb_xy: List[Point],
    jb_types: List[Set[str]],
    comp: List[int],
    *,
    walls: List[List[Point]],
    grid_px: float,
    clearance_px: float,
    k_attach: int = 8,
    node_limit: int = 20000,
    deadline_s: float = 0.30,
) -> Tuple[List[List[Point]], List[Tuple[int, int, Point, Set[str]]]]:
    """
    改进版逻辑：

    1. 先按原来算法把所有 JB 接成一棵树，得到:
         - paths: 每一步新增的接入路径（几何形状）
         - joins_basic: 对应的 (u_idx, v_idx, join_point)（不再在这里带回路信息）
    2. 用这些 paths 在网格上构建一个无向图（trunk segment 的拓扑结构）。
    3. 对 comp 里的每个 JB，沿 “JB -> Panel” 做一次 BFS，
       把该 JB 的回路集合记到它经过的每一个 segment 上（edges_circuits）。
       这样每条 segment 知道自己实际承载了多少个不同回路。
    4. 对每条 path：
       - 看这条 path 上所有 segment，在这些 segment 里挑
         “回路数量最多的那一段”的回路集合 best_set，
       - 把 best_set 作为这条 path 的 types_on_path。
         ==> 线宽由该 path 上最胖的那一段决定，
             所以 main bus 中间段不会被缩成 (3)，
             只有真正末端的小尾巴才是 (3)。
    """
    from collections import defaultdict, deque

    def manhattan(a: Point, b: Point) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    panel_grid = snap(panel_coord, grid_px)

    # ------- 第一步：按旧逻辑构建 Steiner 树（不在这里算 types） -------
    tree_nodes: List[Point] = [panel_grid]
    # 节点上预估的回路集合（只用于控制“同一路径上最多 3 个回路”的约束）
    node_types: Dict[Point, Set[str]] = {panel_grid: set()}

    paths: List[List[Point]] = []  # 每次新增的 trunk path（几何）
    joins_basic: List[Tuple[int, int, Point]] = []  # (u_idx, v_idx, join_point)
    path_jb_index: List[int] = []  # paths[k] 是哪个 JB 的接入路径

    remaining: Set[int] = set(comp)

    def jb_node(j: int) -> Point:
        # 比较距离时用 snap 后坐标更合理
        return snap(jb_xy[j], grid_px)

    # 1) 先接最近的 JB（以 Panel 为起点）
    first = min(remaining, key=lambda j: manhattan(panel_coord, jb_xy[j]))
    path0 = route_rect_with_walls(
        panel_grid,
        jb_xy[first],
        walls=walls,
        grid_px=grid_px,
        clearance_px=clearance_px,
    )
    S0 = set(jb_types[first])
    for p in _sample_nodes_on_polyline(path0, grid_px):
        if p not in node_types:
            node_types[p] = set(S0)
            tree_nodes.append(p)
        else:
            node_types[p] |= S0
            if len(node_types[p]) > 3:
                # 超过 3 种只保留前 3 种（和你原来的保持一致）
                node_types[p] = set(list(node_types[p])[:3])

    paths.append(path0)
    joins_basic.append((0, first + 1, path0[-1]))  # 先不放 types
    path_jb_index.append(first)
    remaining.remove(first)

    # 2) 依次把剩余 JB 接入已有的树
    while remaining:
        best = None
        best_j = None
        best_to = None
        best_path = None
        best_types = None

        for j in list(remaining):
            Sj = set(jb_types[j])  # 当前 JB 自己的回路集合
            # 候选接入点：距离该 JB 最近的若干 tree_nodes
            cand_nodes = (
                sorted(tree_nodes, key=lambda p: manhattan(jb_xy[j], p))[:k_attach]
                if len(tree_nodes) > k_attach
                else tree_nodes
            )
            for t in cand_nodes:
                # 预估如果接在 t 上，这里会有多少种回路
                U = set(node_types.get(t, set())) | Sj
                if len(U) > 3:
                    # 超过 3 种，直接跳过这个接入点
                    continue
                path = route_rect_with_walls(
                    jb_xy[j],
                    t,
                    walls=walls,
                    grid_px=grid_px,
                    clearance_px=clearance_px,
                    node_limit=node_limit,
                    deadline_s=deadline_s,
                )
                if path is None:
                    continue
                L = rect_path_length(path)
                if (best is None) or (L < best):
                    best = L
                    best_j = j
                    best_to = t
                    best_path = path
                    best_types = U

        # 如果完全找不到可行接入（很极端的兜底）
        if best_path is None:
            j = remaining.pop()
            Sj = set(jb_types[j])
            path = route_rect_with_walls(
                panel_grid,
                jb_xy[j],
                walls=walls,
                grid_px=grid_px,
                clearance_px=clearance_px,
                node_limit=node_limit,
                deadline_s=deadline_s,
            )
            for p in _sample_nodes_on_polyline(path, grid_px):
                if p not in node_types:
                    node_types[p] = set(Sj)
                    tree_nodes.append(p)
                else:
                    node_types[p] |= Sj
                    if len(node_types[p]) > 3:
                        node_types[p] = set(list(node_types[p])[:3])
            paths.append(path)
            joins_basic.append((0, j + 1, path[-1]))
            path_jb_index.append(j)
            continue

        # 把 best_j 按 best_path 接入 best_to
        for p in _sample_nodes_on_polyline(best_path, grid_px):
            if p not in node_types:
                node_types[p] = set(best_types)
                tree_nodes.append(p)
            else:
                node_types[p] |= best_types
                if len(node_types[p]) > 3:
                    node_types[p] = set(list(node_types[p])[:3])

        # 计算 attach_idx：0=Panel，>0=JB+1，-1=接在树中间点
        attach_idx = -1
        if best_to == panel_grid:
            attach_idx = 0
        else:
            for j0 in comp:
                if best_to == jb_node(j0):
                    attach_idx = j0 + 1
                    break

        paths.append(best_path)
        joins_basic.append((attach_idx, best_j + 1, best_path[-1]))
        path_jb_index.append(best_j)
        remaining.remove(best_j)

    # ------- 第二步：把所有 trunk path 离散成网格节点，构建 adjacency -------
    def _path_nodes(poly: List[Point]) -> List[Point]:
        pts = _sample_nodes_on_polyline(poly, grid_px)
        out: List[Point] = []
        last: Optional[Point] = None
        for p in pts:
            q = snap(p, grid_px)
            if (last is None) or (q != last):
                out.append(q)
                last = q
        return out

    adj: Dict[Point, Set[Point]] = defaultdict(set)
    path_nodes_list: List[List[Point]] = []

    for poly in paths:
        nodes = _path_nodes(poly)
        path_nodes_list.append(nodes)
        for a, b in zip(nodes, nodes[1:]):
            adj[a].add(b)
            adj[b].add(a)

    # ★ 兜底：如果 panel_grid 不在图里，就接到最近的一个节点上
    if panel_grid not in adj and adj:
        nearest = min(adj.keys(), key=lambda p: manhattan(p, panel_grid))
        adj[panel_grid].add(nearest)
        adj[nearest].add(panel_grid)
    panel_node = panel_grid

    def bfs_path(start: Point, goal: Point) -> Optional[List[Point]]:
        """在 trunk 图上找 start -> goal 的最短路径（曼哈顿步数），用来确定 JB->Panel 的实际经过 segment。"""
        if start == goal:
            return [start]
        dq = deque([start])
        prev: Dict[Point, Optional[Point]] = {start: None}
        while dq:
            u = dq.popleft()
            if u == goal:
                break
            for v in adj.get(u, ()):
                if v not in prev:
                    prev[v] = u
                    dq.append(v)
        if goal not in prev:
            return None
        path: List[Point] = []
        cur: Optional[Point] = goal
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    def _norm_seg(a: Point, b: Point) -> Tuple[Point, Point]:
        """segment 用无向 key 存：两端点按大小排序，这样 AB 和 BA 视为同一条段。"""
        return (a, b) if a <= b else (b, a)

    # ------- 第三步：对每个 JB，把它的回路打到 JB->Panel 的每条 segment 上 -------
    edges_circuits: Dict[Tuple[Point, Point], Set[str]] = defaultdict(set)

    for j in comp:
        circuits_j = {normalize_device_label(l) for l in jb_types[j]}
        if not circuits_j:
            continue
        start = snap(jb_xy[j], grid_px)
        if start not in adj:
            if adj:
                nearest = min(adj.keys(), key=lambda p: manhattan(p, start))

                start = nearest
            else:

                continue
        node_path = bfs_path(start, panel_node)
        if not node_path or len(node_path) < 2:

            continue
        for a, b in zip(node_path, node_path[1:]):
            key = _norm_seg(a, b)
            edges_circuits[key].update(circuits_j)

    # ------- 第四步：给每条 path 选合适的回路集合作为 types_on_path -------
    joins: List[Tuple[int, int, Point, Set[str]]] = []

    for (u_idx, v_idx, join_p), nodes_seq, jb_idx in zip(
        joins_basic, path_nodes_list, path_jb_index
    ):
        # 1) 先按原来的思路：看整条 path 上 segment 的统计，找“最胖”的那一段
        edges = [_norm_seg(a, b) for a, b in zip(nodes_seq, nodes_seq[1:])]
        best_set: Set[str] = set()
        best_size = -1
        for e in edges:
            s = edges_circuits.get(e)
            if not s:
                continue
            if len(s) > best_size:
                best_size = len(s)
                best_set = set(s)

        # 如果 BFS 没有给到任何信息，就退回用这个 JB 自己的回路
        if not best_set:
            best_set = {normalize_device_label(l) for l in jb_types[jb_idx]}

        # 2) 关键补丁：对“非 Panel 起点”的 path，限制成端点可能承载的回路
        #
        # v_idx 一定是 JB（>0）
        endpoint_union: Set[str] = set()
        if v_idx > 0:
            endpoint_union |= {normalize_device_label(l) for l in jb_types[v_idx - 1]}
        # u_idx 可能是 Panel(0) / JB(>0) / -1(TRUNK)
        if u_idx > 0:
            endpoint_union |= {normalize_device_label(l) for l in jb_types[u_idx - 1]}

        # 只有当起点不是 Panel 时才做端点过滤：
        #   - Panel→某 JB 那条 path 保持原来的 “最胖段” 逻辑，保证 panel 端的 main bus 足够粗
        #   - JB↔JB / JB↔Steiner 的 path，会把“跟端点无关的别的回路”剔掉，
        #     例如 JBR10(10)↔JBR11(10) 不会再被别的 family 污染成 {10,27}
        if u_idx != 0 and endpoint_union:
            filtered = best_set & endpoint_union
            if filtered:
                best_set = filtered
            else:
                # 理论上不会发生；保险起见，如果被全剔掉了，就直接用端点并集
                best_set = set(endpoint_union)

        joins.append((u_idx, v_idx, join_p, best_set))
    return paths, joins


# ==================== 分割线左右判定 ====================


def _point_side_of_line(
    p: Point, a: Point, b: Point, *, on_line_tol_px: float = 2.0
) -> str:
    (px, py) = p
    (ax, ay) = a
    (bx, by) = b
    vx, vy = (bx - ax), (by - ay)
    wx, wy = (px - ax), (py - ay)
    cross = vx * wy - vy * wx
    seg_len = math.hypot(vx, vy) or 1.0
    dist = abs(cross) / seg_len
    if dist <= on_line_tol_px:
        return "ON"
    return "L" if cross > 0 else "R"


def _split_dets_by_line(
    dets: List[Dict],
    divider_line: Tuple[Point, Point],
    left_panel: Point,
    right_panel: Point,
    *,
    min_conf: float,
    keep_labels: Set[str],
    on_line_tol_px: float = 2.0,
) -> Tuple[List[Dict], List[Dict]]:
    a, b = divider_line
    left_dets = []
    right_dets = []
    for d in dets:
        if d.get("confidence", 1.0) < min_conf:
            continue
        lab = d.get("label")
        if lab not in keep_labels:
            continue
        cx = 0.5 * (d["x1"] + d["x2"])
        cy = 0.5 * (d["y1"] + d["y2"])
        side = _point_side_of_line((cx, cy), a, b, on_line_tol_px=on_line_tol_px)
        if side == "L":
            left_dets.append(d)
        elif side == "R":
            right_dets.append(d)
        else:
            d_left = manhattan((cx, cy), left_panel)
            d_right = manhattan((cx, cy), right_panel)
            (left_dets if d_left <= d_right else right_dets).append(d)
    return left_dets, right_dets


# ==================== 双 Panel（主流程） ====================


def _draw_side(
    page: fitz.Page,
    dets_side: List[Dict],
    *,
    panel_coord: Point,
    color,
    capacity: int,
    eps_px: float,
    grid_px: float,
    merge_eps: float,
    min_conf: float,
    keep_labels: Set[str],
    jb_radius: float,
    dev_radius: float,
    panel_radius: float,
    wire_width: float,
    walls: Optional[List[List[Point]]],
    clearance_px: float,
    color_map: Optional[Dict[str, Tuple[float, float, float]]],
    priority_families: Optional[List[Iterable[str]]],
    jb_prefix: str,
    lane_gap: float,
    lane_mode: str,  # "axis" | "diag"
    allow_branch_overlap: bool,  # True=支线可重叠（但不会和主干重叠）
    allow_trunk_overlap: bool,  # True=主干可重叠
    mark_virtual_steiner: bool = True,
):
    # 当前侧没有设备：只画 Panel 圈
    if not dets_side:
        draw_circle_annot(
            page,
            panel_coord,
            r=panel_radius,
            fill_color=color,
            width=0.4,
            title="PANEL",
        )
        return {
            "num_jb": 0,
            "dev_count": 0,
            "jb_label_sets": [],
            "families": [],
            "trunk_edges": 0,
        }

    # 自动放置 JB
    jb_xy, dev2jb, groups, dev_coords, jb_label_sets, family_id_of_label, families = (
        auto_place_junction_boxes(
            dets_side,
            capacity=capacity,
            eps_px=eps_px,
            grid_px=grid_px,
            merge_eps=merge_eps,
            min_conf=min_conf,
            keep_labels=keep_labels,
            return_coords=True,
            priority_families=priority_families,
            walls=walls,
            clearance_px=clearance_px,
        )
    )

    # JB 标注（含类型）
    for j, pt in enumerate(jb_xy):
        types_str = ",".join(sorted(list(jb_label_sets[j])))
        draw_circle_annot(
            page,
            pt,
            r=jb_radius,
            fill_color=None,
            width=1.1,
            title="JB",
            contents=f"{jb_prefix}{j} | types:[{types_str}]",
        )

    # 设备点（kept_idx 顺序与 dev_coords 一一对应）
    kept_idx = [
        i
        for i, d in enumerate(dets_side)
        if d.get("confidence", 1.0) >= min_conf and d.get("label") in keep_labels
    ]
    kept_centres = dev_coords
    for local_i, (x, y) in enumerate(kept_centres):
        lab_i = dets_side[kept_idx[local_i]]["label"]
        col_i = color_map.get(lab_i) if color_map else color
        draw_circle_annot(
            page,
            (x, y),
            r=dev_radius,
            fill_color=col_i,
            width=0.2,
            title="DEVICE",
            contents=f"Dev{local_i} ({lab_i})",
        )

    # --- 并排走线：本侧共用一个 lane_map（主干和支线都用它来避免重叠） ---
    lane_map: Dict[Tuple[Point, Point], Set[int]] = defaultdict(set)

    # ======= 先画史坦纳主干（保证路径 ≤3 种），主干先占 lane =======
    comps = components_by_family(
        jb_xy,
        jb_label_sets,
        max_types=3,
        family_id_of_label=family_id_of_label,
        families=families,
    )

    def panel_name():
        return (
            "PANEL_L"
            if jb_prefix == "JBL"
            else ("PANEL_R" if jb_prefix == "JBR" else "PANEL")
        )

    def node_with_types_idx(idx: int) -> str:
        if idx == 0:
            return panel_name()
        j = idx - 1
        types = ",".join(sorted(list(jb_label_sets[j])))
        return f"{jb_prefix}{j}({types})"

    trunk_edges_cnt = 0

    panel_node = snap(panel_coord, grid_px)

    for comp in comps:
        # 1. 先用原函数生成 Steiner 主干几何（不再用它决定线宽）
        trunk_paths, joins = build_steiner_trunk_paths(
            panel_coord,
            jb_xy,
            jb_label_sets,
            comp,
            walls=walls or [],
            grid_px=grid_px,
            clearance_px=clearance_px,
            k_attach=8,
            node_limit=20000,
            deadline_s=0.30,
        )

        # 2. 这个 family 里：JB 节点 -> 该节点下所有 JB 的回路集合
        node_to_jb_indices: Dict[Point, List[int]] = defaultdict(list)
        node_circuits: Dict[Point, Set[str]] = defaultdict(set)
        for j in comp:
            n = snap(jb_xy[j], grid_px)
            node_to_jb_indices[n].append(j)
            node_circuits[n].update(
                normalize_device_label(lab) for lab in jb_label_sets[j]
            )

        # 3. 把所有 trunk path 拆成“网格节点 + 邻接表”
        def _path_nodes(poly: List[Point]) -> List[Point]:
            pts = _sample_nodes_on_polyline(poly, grid_px)
            out: List[Point] = []
            last: Optional[Point] = None
            for p in pts:
                q = snap(p, grid_px)
                if (last is None) or (q != last):
                    out.append(q)
                    last = q
            return out

        adj: Dict[Point, Set[Point]] = defaultdict(set)
        for poly in trunk_paths:
            nodes_seq = _path_nodes(poly)
            for a, b in zip(nodes_seq, nodes_seq[1:]):
                adj[a].add(b)
                adj[b].add(a)

        # 4. 以 panel_node 为根建一棵树（BFS），得到 parent / children
        from collections import deque, defaultdict as ddict

        parent: Dict[Point, Optional[Point]] = {panel_node: None}
        order: List[Point] = [panel_node]
        dq = deque([panel_node])
        while dq:
            u = dq.popleft()
            for v in adj.get(u, ()):
                if v not in parent:
                    parent[v] = u
                    dq.append(v)
                    order.append(v)

        children: Dict[Point, List[Point]] = ddict(list)
        for v, p in parent.items():
            if p is not None:
                children[p].append(v)

        # 5. 自底向上算每个节点的“子树回路并集”，再得到每条边的回路集合
        circuits_subtree: Dict[Point, Set[str]] = {}
        for v in reversed(order):
            S = set(node_circuits.get(v, set()))
            for c in children.get(v, ()):
                S |= circuits_subtree[c]
            circuits_subtree[v] = S

        edge_circuits: Dict[Tuple[Point, Point], Set[str]] = {}
        for v, p in parent.items():
            if p is None:
                continue
            key = _norm_seg(p, v)
            edge_circuits[key] = circuits_subtree[v]

        # 6. 给节点起个名字（方便 contents 里写 “谁到谁”）
        def node_name(pt: Point) -> str:
            if pt == panel_node:
                return panel_name()
            jb_list = node_to_jb_indices.get(pt)
            if jb_list:
                if len(jb_list) == 1:
                    j_idx = jb_list[0]
                    types = ",".join(sorted(jb_label_sets[j_idx]))
                    return f"{jb_prefix}{j_idx}({types})"
                else:
                    parts = []
                    for j_idx in jb_list:
                        types = ",".join(sorted(jb_label_sets[j_idx]))
                        parts.append(f"{jb_prefix}{j_idx}({types})")
                    return " & ".join(parts)
            return "TRUNK"

        # 7. 真正画主干：
        #    先在树上把“回路集合完全相同”的连续边合并成一条 polyline，
        #    再按每条 polyline 的 circuits_set 决定线宽。
        trunk_chains = build_trunk_chains_from_tree(
            adj,
            edge_circuits,
            root=panel_node,  # 在 Panel 节点强制断开，避免跨 Panel 合并
        )

        for nodes_seq, circuits in trunk_chains:
            types = {
                normalize_device_label(t) for t in (circuits or set()) if t is not None
            }
            type_count = len(types)
            if type_count <= 0:
                continue  # 没回路就不画

            # chain 本身就是顺序的节点列表，把它 snap 成 polyline
            poly = snap_poly_to_grid(nodes_seq, grid_px)

            lane = choose_lane_for_poly(
                poly,
                lane_map,
                grid_px,
                allow_overlap=allow_trunk_overlap,
            )
            if lane_mode == "axis":
                poly_o = offset_poly_axiswise(poly, lane, lane_gap)
            else:
                poly_o = offset_poly_diag(poly, lane, lane_gap)

            subj = mainbus_subject_for_type_count(type_count)
            trunk_width = mainbus_width_for_type_count(type_count)

            L_pts = rect_path_length(poly)
            L_str = format_length_ft_in(L_pts, inch_precision=0)

            u_name = node_name(poly[0])
            v_name = node_name(poly[-1])

            draw_polyline_annot(
                page,
                poly_o,
                width=trunk_width,
                stroke_color=color,
                title=subj,
                contents=f"{u_name} -> {v_name} | L={L_str}",
            )

            # 这里按“包含的边数”统计一下主干段数（可选）
            trunk_edges_cnt += max(len(poly) - 1, 1)

        # 8. 虚拟 JB（Steiner 点）的小圈仍然用原来的 joins 逻辑，只是不要再画主干线了
        connected_anchors: List[int] = []
        for poly, jinfo in zip(trunk_paths, joins):
            u_idx, v_idx, join_point, types_on_path = jinfo

            if mark_virtual_steiner and u_idx < 0:

                def anchor_point(idx_anchor: int) -> Point:
                    if idx_anchor == 0:
                        return snap(panel_coord, grid_px)
                    else:
                        j = idx_anchor - 1
                        return snap(jb_xy[j], grid_px)

                if not connected_anchors:
                    merged_label = (
                        f"merge of {panel_name()} & {node_with_types_idx(v_idx)}"
                    )
                else:
                    cand = connected_anchors + [v_idx]
                    best_pair = None
                    best_cost = None
                    for i in range(len(cand)):
                        for j in range(i + 1, len(cand)):
                            pa = anchor_point(cand[i])
                            pb = anchor_point(cand[j])
                            cost = manhattan(pa, join_point) + manhattan(pb, join_point)
                            if best_cost is None or cost < best_cost:
                                best_cost = cost
                                best_pair = (cand[i], cand[j])
                    if best_pair:
                        a_idx, b_idx = best_pair
                        a_name = node_with_types_idx(a_idx)
                        b_name = node_with_types_idx(b_idx)
                        merged_label = f"merge of {a_name} & {b_name}"
                    else:
                        merged_label = f"merge near {node_with_types_idx(v_idx)}"

                draw_circle_annot(
                    page,
                    join_point,
                    r=jb_radius * 0.6,
                    fill_color=None,
                    width=0.9,
                    title="JB",
                    contents=f"{jb_prefix}S | {merged_label}",
                )

            if u_idx >= 0 and u_idx not in connected_anchors:
                connected_anchors.append(u_idx)
            if v_idx not in connected_anchors:
                connected_anchors.append(v_idx)

    # ======= 再画设备→JB 支线，并且永远不与主干重叠 =======
    # 设备→JB（subject 固定为 (3)）
    local2jb: Dict[int, int] = {}
    for gidx, jb in dev2jb.items():
        # dev2jb 的 key 是 dets_side 的索引（raw_idx）
        local = kept_idx.index(gidx)
        local2jb[local] = jb

    for local_i, (x, y) in enumerate(kept_centres):
        if local_i not in local2jb:
            continue
        jb_idx = local2jb[local_i]
        jx, jy = jb_xy[jb_idx]

        poly = route_rect_with_walls(
            (x, y),
            (jx, jy),
            walls=walls or [],
            grid_px=grid_px,
            clearance_px=clearance_px,
        )
        poly = snap_poly_to_grid(poly, grid_px)

        # 支线与主干不允许重叠：
        #   - 永远参考 lane_map（里面已经有主干占用信息）
        #   - allow_branch_overlap=True：支线之间可以互相重叠（不写回 lane_map）
        #   - allow_branch_overlap=False：支线之间也不重叠（写回 lane_map）
        lane = choose_lane_for_poly(
            poly,
            lane_map,
            grid_px,
            allow_overlap=False,  # 一定要考虑已有占道（至少避开主干）
            update_map=not allow_branch_overlap,  # 控制支线之间是否互相占 lane
        )

        if lane_mode == "axis":
            poly_o = offset_poly_axiswise(poly, lane, lane_gap)
        else:
            poly_o = offset_poly_diag(poly, lane, lane_gap)

        lab_i = dets_side[kept_idx[local_i]]["label"]
        col_i = color_map.get(lab_i) if color_map else color
        L_pts = rect_path_length(poly)
        L_str = format_length_ft_in(L_pts, inch_precision=0)

        # draw_polyline_annot(
        #     page,
        #     poly_o,
        #     width=wire_width,
        #     stroke_color=col_i,
        #     title=BRANCH_SUBJECT,
        #     contents=f"Dev{local_i}({lab_i})->{jb_prefix}{jb_idx} | L={L_str}",
        # )

    # Panel 标注
    draw_circle_annot(
        page,
        panel_coord,
        r=panel_radius,
        fill_color=color,
        width=0.4,
        title="PANEL",
        contents=f"{'LEFT' if jb_prefix=='JBL' else 'RIGHT'}",
    )

    return {
        "num_jb": len(jb_xy),
        "dev_count": len(kept_centres),
        "jb_label_sets": [sorted(list(s)) for s in jb_label_sets],
        "families": [sorted(list(s)) for s in families],
        "trunk_edges": trunk_edges_cnt,
    }


def route_with_jb_strategy_dual_panels(
    input_pdf: str,
    output_pdf: str,
    page_index: int,
    dets: List[Dict],
    *,
    left_panel_coord: Point,
    right_panel_coord: Point,
    divider_line: Tuple[Point, Point],
    color=(1, 0, 0),
    capacity=8,
    eps_px=400,
    grid_px=50,
    merge_eps=50,
    min_conf=0.30,
    keep_labels: Optional[Iterable[str]] = None,
    jb_radius=10.0,
    dev_radius=6.0,
    panel_radius=12.0,
    wire_width=1.0,
    walls: Optional[List[List[Point]]] = None,
    clearance_px: float = 0.0,
    color_map: Optional[Dict[str, Tuple[float, float, float]]] = None,
    priority_families: Optional[List[Iterable[str]]] = None,
    on_line_tol_px: float = 2.0,
    lane_gap: float = 8.0,  # 并排间距
    lane_mode: str = "axis",  # "axis"（正交友好）或 "diag"
    allow_branch_overlap: bool = True,  # 支线默认允许重叠
    allow_trunk_overlap: bool = False,  # 主干默认不允许重叠
):
    doc = fitz.open(input_pdf)
    out = fitz.open()
    out.insert_pdf(doc)
    doc.close()
    page = out[page_index]
    if keep_labels is None:
        all_labels = {d["label"] for d in dets if d.get("label") is not None}
        keep_labels = {l for l in all_labels if l != "PANEL"}
    keep_labels = set(keep_labels)
    left_dets, right_dets = _split_dets_by_line(
        dets,
        divider_line,
        left_panel_coord,
        right_panel_coord,
        min_conf=min_conf,
        keep_labels=keep_labels,
        on_line_tol_px=on_line_tol_px,
    )
    left_info = _draw_side(
        page,
        left_dets,
        panel_coord=left_panel_coord,
        color=color,
        capacity=capacity,
        eps_px=eps_px,
        grid_px=grid_px,
        merge_eps=merge_eps,
        min_conf=min_conf,
        keep_labels=keep_labels,
        jb_radius=jb_radius,
        dev_radius=dev_radius,
        panel_radius=panel_radius,
        wire_width=wire_width,
        walls=walls,
        clearance_px=clearance_px,
        color_map=color_map,
        priority_families=priority_families,
        jb_prefix="JBL",
        lane_gap=lane_gap,
        lane_mode=lane_mode,
        allow_branch_overlap=allow_branch_overlap,
        allow_trunk_overlap=allow_trunk_overlap,
        mark_virtual_steiner=True,
    )
    right_info = _draw_side(
        page,
        right_dets,
        panel_coord=right_panel_coord,
        color=color,
        capacity=capacity,
        eps_px=eps_px,
        grid_px=grid_px,
        merge_eps=merge_eps,
        min_conf=min_conf,
        keep_labels=keep_labels,
        jb_radius=jb_radius,
        dev_radius=dev_radius,
        panel_radius=panel_radius,
        wire_width=wire_width,
        walls=walls,
        clearance_px=clearance_px,
        color_map=color_map,
        priority_families=priority_families,
        jb_prefix="JBR",
        lane_gap=lane_gap,
        lane_mode=lane_mode,
        allow_branch_overlap=allow_branch_overlap,
        allow_trunk_overlap=allow_trunk_overlap,
        mark_virtual_steiner=True,
    )
    out.save(output_pdf)
    out.close()
    return {
        "left": left_info,
        "right": right_info,
        "left_count_dets": len(left_dets),
        "right_count_dets": len(right_dets),
    }


# ==================== 示例入口 ====================

if __name__ == "__main__":
    # 这里用你上一条消息中的 L2_A 示例；按需替换
    file_name = "pdf_files/L2_B"
    input_pdf = f"{file_name}.pdf"
    output_pdf_route = f"{file_name}_route_dual_panels.pdf"
    PAGE_INDEX = 0
    DEVICE_LABEL = ["16", "14", "12", "26", "8", "22", "29", "20"]
    color_map = pick_device_colors(DEVICE_LABEL)

    LEFT_PANEL: Point = (1745.0, 2526.0)
    RIGHT_PANEL: Point = (1186.0, 1030.0)
    DIVIDER_LINE = ((504.26, 1624.05), (2069.37, 1624.05))
    WALLS = [
        [
            (1798.677978515625, 2501.622314453125),
            (1798.677978515625, 2606.0419921875),
            (1705.4019775390625, 2606.0419921875),
            (1705.4019775390625, 2464.47705078125),
            (1799.0909423828125, 2464.47705078125),
        ],
        [
            (1797.791015625, 2302.97412109375),
            (1797.791015625, 2070.4638671875),
            (1705.0760498046875, 2070.4638671875),
            (1705.0760498046875, 2357.8798828125),
            (1798.5040283203125, 2357.8798828125),
        ],
        [(1705.7919921875, 2070.9521484375), (1392.4639892578125, 2070.9521484375)],
        [
            (1182.625, 962.952880859375),
            (1231.1610107421875, 962.952880859375),
            (1231.1610107421875, 1253.1820068359375),
            (1145.9749755859375, 1253.1820068359375),
            (1145.9749755859375, 961.9619140625),
        ],
        [
            (1146.489990234375, 1248.9630126953125),
            (921.7600708007812, 1248.9630126953125),
        ],
        [
            (1348.33203125, 606.00390625),
            (1348.33203125, 682.701904296875),
            (1231.3929443359375, 682.701904296875),
            (1231.3929443359375, 852.60791015625),
            (1198.375, 852.60791015625),
            (1198.375, 903.510986328125),
        ],
        [
            (1229.6729736328125, 718.1279296875),
            (1137.842041015625, 718.1279296875),
            (1137.842041015625, 853.639892578125),
            (1150.5670166015625, 853.639892578125),
        ],
    ]
    CLEARANCE = 3.0

    doc = fitz.open(input_pdf)
    page = doc[PAGE_INDEX]
    dets_all = build_dets_from_pdf_text_multi(page, DEVICE_LABEL)
    doc.close()

    info = route_with_jb_strategy_dual_panels(
        input_pdf=input_pdf,
        output_pdf=output_pdf_route,
        page_index=PAGE_INDEX,
        dets=dets_all,
        left_panel_coord=LEFT_PANEL,
        right_panel_coord=RIGHT_PANEL,
        divider_line=DIVIDER_LINE,
        color=(1, 0, 0),
        capacity=8,
        eps_px=400,
        grid_px=50,
        merge_eps=50,
        min_conf=0.30,
        keep_labels=set(DEVICE_LABEL),
        jb_radius=10.0,
        dev_radius=6.0,
        panel_radius=12.0,
        wire_width=3.0,
        walls=WALLS,
        clearance_px=CLEARANCE,
        color_map=color_map,
        priority_families=[{"14"}],
        on_line_tol_px=2.0,
        lane_gap=8.0,  # 并排间距
        lane_mode="diag",  # "axis"（正交友好）或 "diag"
        allow_branch_overlap=True,  # 支线允许重叠（默认）
        allow_trunk_overlap=False,  # 主干不允许重叠（默认）
    )
    print("Dual-panels routing summary:", info)
