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

# Point = Tuple[float, float]
Edge = Tuple[int, int]

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
    t = 0.0 if denom == 0 else clamp01((ux * vx + uy * vy) / denom)
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


# ==================== 家族（≤3） ====================


def build_disjoint_families(
    dets: List[Dict],
    keep_labels: Iterable[str],
    *,
    neighbor_eps_px: float = 400.0,
    min_conf: float = 0.30,
    priority_families: Optional[List[Iterable[str]]] = None,
    max_family_size: int = 3,
) -> Tuple[Dict[str, int], List[Set[str]]]:
    keep_labels = set(keep_labels)
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
    idx_of_label = {lab: i for i, lab in enumerate(labels)}
    parent = list(range(len(labels)))
    size = [1] * len(labels)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return rx
        if size[rx] < size[ry]:
            rx, ry = ry, rx
        if size[rx] + size[ry] > max_family_size:
            return rx
        parent[ry] = rx
        size[rx] += size[ry]
        return rx

    if priority_families:
        for fam in priority_families:
            fam = sorted(set(fam) & set(labels))
            if not fam:
                continue
            base = idx_of_label[fam[0]]
            for lab in fam[1:]:
                base = union(base, idx_of_label[lab])
    for (a, b), _w in co.most_common():
        if a not in idx_of_label or b not in idx_of_label:
            continue
        union(idx_of_label[a], idx_of_label[b])
    fam_id_map = {}
    families = []
    for lab, i in idx_of_label.items():
        r = find(i)
        if r not in fam_id_map:
            fam_id_map[r] = len(families)
            families.append(set())
        families[fam_id_map[r]].add(lab)
    family_id_of_label = {lab: fam_id_map[find(idx_of_label[lab])] for lab in labels}
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
):
    if keep_labels is None:
        all_labels = {d["label"] for d in dets if d.get("label") is not None}
        keep_labels = {l for l in all_labels if l != "PANEL"}
    keep_labels = set(keep_labels)
    family_id_of_label, families = build_disjoint_families(
        dets,
        keep_labels,
        neighbor_eps_px=eps_px,
        min_conf=min_conf,
        priority_families=priority_families,
        max_family_size=3,
    )
    raw_pts = []
    raw_idx = []
    raw_labs = []
    for i, d in enumerate(dets):
        if d.get("confidence", 1.0) < min_conf:
            continue
        lab = d.get("label")
        if lab not in keep_labels:
            continue
        cx = 0.5 * (d["x1"] + d["x2"])
        cy = 0.5 * (d["y1"] + d["y2"])
        raw_pts.append((cx, cy))
        raw_idx.append(i)
        raw_labs.append(lab)
    if not raw_pts:
        raise ValueError("No valid devices after filtering")

    dev_points = []
    dev_label_sets = []
    dev_counts = []
    dev_point_to_global_idxs = []
    by_label = defaultdict(list)
    for k, lab in enumerate(raw_labs):
        by_label[lab].append(k)
    for lab, idxs in by_label.items():
        pts_lab = np.asarray([raw_pts[k] for k in idxs], float)
        if len(pts_lab) == 0:
            continue
        labels_lab = DBSCAN(eps=merge_eps, min_samples=1).fit_predict(pts_lab)
        for val in np.unique(labels_lab):
            sub = np.where(labels_lab == val)[0]
            members = [idxs[s] for s in sub]
            p = pts_lab[sub].mean(axis=0)
            dev_points.append((float(p[0]), float(p[1])))
            dev_label_sets.append({lab})
            dev_counts.append(len(members))
            dev_point_to_global_idxs.append([raw_idx[m] for m in members])
    dev_points_np = np.asarray(dev_points, float)

    JB_centers = []
    for p in dev_points_np:
        snapped = snap((float(p[0]), float(p[1])), grid_px)
        JB_centers.append(
            nudge_point_off_walls(
                snapped, walls=walls or [], grid_px=grid_px, clearance_px=clearance_px
            )
        )
    JB_label_sets = [set(s) for s in dev_label_sets]
    JB_counts = list(dev_counts)
    JB_members = [[i] for i in range(len(dev_points))]
    active = [True] * len(JB_centers)

    def same_family(L: Set[str]) -> bool:
        if not L:
            return True
        fids = {family_id_of_label.get(t) for t in L if t in family_id_of_label}
        fids.discard(None)
        if len(fids) != 1:
            return False
        fid = next(iter(fids))
        return L.issubset(families[fid])

    def can_merge(i, j):
        if i == j or (not active[i]) or (not active[j]):
            return False
        if manhattan(JB_centers[i], JB_centers[j]) > eps_px:
            return False
        if JB_counts[i] + JB_counts[j] > capacity:
            return False
        L = JB_label_sets[i] | JB_label_sets[j]
        if len(L) > 3:
            return False
        if not same_family(L):
            return False
        return True

    def merge_score(i, j):
        Li, Lj = JB_label_sets[i], JB_label_sets[j]
        inter = len(Li & Lj)
        new_added = len((Li | Lj)) - max(len(Li), len(Lj))
        dist = manhattan(JB_centers[i], JB_centers[j])
        return (inter, -new_added, -dist)

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
        new_members = JB_members[i] + JB_members[j]
        JB_members[i] = new_members
        JB_counts[i] += JB_counts[j]
        JB_label_sets[i] |= JB_label_sets[j]
        new_center = tuple(
            np.asarray([dev_points_np[k] for k in new_members], float).mean(axis=0)
        )
        snapped = snap(new_center, grid_px)
        JB_centers[i] = nudge_point_off_walls(
            snapped, walls=walls or [], grid_px=grid_px, clearance_px=clearance_px
        )
        active[j] = False

    jb_xy = []
    jb_label_sets_out = []
    jb_members_out = []
    for old_i, on in enumerate(active):
        if not on:
            continue
        jb_xy.append(JB_centers[old_i])
        jb_label_sets_out.append(JB_label_sets[old_i])
        jb_members_out.append(JB_members[old_i])

    dev2jb = {}
    groups = defaultdict(list)
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


# ==================== 注释绘制 ====================


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


# -*- coding: utf-8 -*-
import fitz  # PyMuPDF
from typing import List, Tuple, Optional, Dict

Point = Tuple[float, float]

# =========================
# 全局缓存（按文档维度缓存一次）
# =========================
TEMPLATE_CACHE: Dict[int, Dict[str, str]] = {}
# 结构示例：
# TEMPLATE_CACHE[id(doc)] = {
#   "subject": "template",
#   "it_value": "/PolyLineDimension",
#   "measure_ref": "123 0 R",          # 永远用 xref 引用；若模板是内联 dict，会在初始化时克隆成一个 xref，后面都用这个引用
#   "page_index": 0,
#   "annot_xref": 45
# }


# -------- 辅助：读取 xref 键 --------
def _xref_get(doc: fitz.Document, xref: int, key: str):
    try:
        t, v = doc.xref_get_key(xref, key)
        return (t or "null"), v
    except Exception:
        return "null", None


# =========================
# 初始化：只在打开文档后调用一次
# =========================
def init_polyline_template_cache(
    doc: fitz.Document,
    template_subject: str = "template",
) -> None:
    """
    扫描全文档，找到 subject=template 的 PolyLine 测量批注，
    把 /IT 与 /Measure 缓存起来（Measure 统一变成 xref 引用，便于快速复用）。
    """
    did = id(doc)

    # 已缓存则直接返回
    if did in TEMPLATE_CACHE and TEMPLATE_CACHE[did].get("subject") == template_subject:
        return

    # 搜索模板
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

    # 读取 /IT
    it_type, it_val = _xref_get(doc, tpl_annot.xref, "IT")
    # 优先希望拿到类似 "/PolyLineDimension" 的 name
    it_value = it_val if it_val else "/PolyLineDimension"  # 没取到就给个常见默认值

    # 读取 /Measure；我们要保证缓存里是 xref 引用（"N 0 R"）
    m_type, m_val = _xref_get(doc, tpl_annot.xref, "Measure")
    if not m_val:
        # 模板上没有 Measure，仍然缓存；画的时候只设置 IT（很多情况下也能工作）
        measure_ref = None
    else:
        if m_type == "xref":
            measure_ref = m_val  # 例如 "123 0 R"
        elif m_type == "dict":
            # 内联字典：克隆成独立对象，得到新的 xref 引用，后面重复使用
            new_m = doc.get_new_xref()
            doc.update_object(new_m, m_val)  # m_val 是 "<< ... >>"
            measure_ref = f"{new_m} 0 R"
        else:
            # 其他罕见情况：先不处理，放 None
            measure_ref = None

    TEMPLATE_CACHE[did] = {
        "subject": template_subject,
        "it_value": it_value,
        "measure_ref": measure_ref,  # 可能为 None
        "page_index": str(tpl_page_idx),
        "annot_xref": str(tpl_annot.xref),
    }


# =========================
# 画：基于缓存的模板创建“可自动测量”的 PolyLine
# =========================
def draw_polyline_annot(
    page: fitz.Page,
    pts: List[Point],
    width: Optional[float] = 0.9,
    stroke_color: Optional[Tuple[float, float, float]] = (1, 0, 0),
    title: str = "WIRE",
    contents: str = "",
    template_subject: str = "template",
    copy_template_appearance: bool = True,  # 是否把模板外观（颜色/线宽/flags/箭头/透明度）也拷过来，再用你的参数覆盖
) -> Optional[fitz.Annot]:
    """
    使用 init_polyline_template_cache 缓存的测量模板属性（/IT + /Measure），
    基于传入顶点创建新的“自动测量”折线，并可覆盖 subject / content / 颜色 / 线宽。
    """
    if len(pts) < 2:
        return None

    doc = page.parent
    did = id(doc)

    # 若未初始化或 subject 不匹配则尝试自动初始化一次
    cache = TEMPLATE_CACHE.get(did)
    if not cache or cache.get("subject") != template_subject:
        init_polyline_template_cache(doc, template_subject)
        cache = TEMPLATE_CACHE.get(did)

    # 创建几何
    plist = [fitz.Point(p[0], p[1]) for p in pts]
    annot = page.add_polyline_annot(plist)

    # 设置 /IT 与 /Measure（保持自动测量）
    it_value = cache.get("it_value")
    if it_value:
        try:
            doc.xref_set_key(annot.xref, "IT", it_value)
        except Exception:
            pass

    measure_ref = cache.get("measure_ref")
    if measure_ref:
        try:
            doc.xref_set_key(annot.xref, "Measure", measure_ref)  # "N 0 R"
        except Exception:
            pass

    # 可选：拷贝模板外观（颜色/线宽/flags/箭头/透明度）
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

    # 用你的参数覆盖
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

    # 标题 / 内容（subject / content）
    try:
        annot.set_info(subject=title)
        if title:
            annot.set_info(title=title)  # 一些阅读器会显示 title
        if contents:
            annot.set_info(content=contents)
    except Exception:
        pass

    # 可选：混合模式
    try:
        annot.set_blendmode("Multiply")
    except Exception:
        pass

    # 关键：测量类不要 update()，避免重建外观导致失去“自动测量”
    # annot.update()

    return annot


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

    # 用“近邻优先”凝聚，阈值放宽为全排序（距离小的优先尝试）
    pairs = []
    for i in range(m):
        for j in range(i + 1, m):
            pairs.append((manhattan(jb_xy[i], jb_xy[j]), i, j))
    pairs.sort(key=lambda t: t[0])
    for _, i, j in pairs:
        union(i, j)
    # 导出分量
    buckets = defaultdict(list)
    for v in range(m):
        buckets[find(v)].append(v)
    return list(buckets.values())


# ==================== 史坦纳主干：最短路生长，去重绘制 ====================


def _sample_nodes_on_polyline(poly: List[Point], grid_px: float) -> List[Point]:
    """按网格步长对已有折线采样，作为可连接的树节点集合。"""
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
    # 去重
    uniq = []
    seen = set()
    for p in out:
        if p in seen:
            continue
        uniq.append(p)
        seen.add(p)
    return uniq


def draw_unique_polyline(
    page: fitz.Page,
    poly: List[Point],
    *,
    stroke_color,
    width: float,
    title: str,
    contents: str,
    drawn_segments: Set[Tuple[Point, Point]],
):
    """只绘制尚未绘制过的线段（避免重复走相同路径）。"""
    if len(poly) < 2:
        return
    cur = []

    def norm_seg(a: Point, b: Point):
        return (a, b) if a <= b else (b, a)

    for a, b in zip(poly, poly[1:]):
        seg = norm_seg(a, b)
        if seg in drawn_segments:
            if len(cur) >= 2:
                draw_polyline_annot(
                    page,
                    cur,
                    width=width,
                    stroke_color=stroke_color,
                    title=title,
                    contents=contents,
                )
            cur = []  # 这段已经画过，断开
        else:
            drawn_segments.add(seg)
            if not cur:
                cur.append(a)
            cur.append(b)
    if len(cur) >= 2:
        draw_polyline_annot(
            page,
            cur,
            width=width,
            stroke_color=stroke_color,
            title=title,
            contents=contents,
        )


def build_steiner_trunk_paths(
    panel_coord: Point,
    jb_xy: List[Point],
    comp: List[int],  # 该分量中的 JB 索引列表
    *,
    walls: List[List[Point]],
    grid_px: float,
    clearance_px: float,
    k_attach: int = 8,  # 每次尝试连接到的最近树节点数量
    node_limit: int = 20000,
    deadline_s: float = 0.30,
) -> Tuple[List[List[Point]], List[Tuple[int, int, Point]]]:
    """
    在一个分量里用“最短路生长”生成主干路径集合。
    返回：
      paths: 由若干折线段组成的列表（每个新增 JB 的接入路径）
      joins: [(u_idx, v_idx, join_point)]，用于标注端点（u/v 是 trunk_points 索引：0=Panel，>0=JB+1）
    """
    # 已有树的节点集（网格点），初始为 Panel
    tree_nodes = [snap(panel_coord, grid_px)]
    paths = []
    joins = []
    remaining = set(comp)

    # 先找到离 Panel 最近的 JB 作为种子
    first = min(remaining, key=lambda j: manhattan(panel_coord, jb_xy[j]))
    path0 = route_rect_with_walls(
        panel_coord,
        jb_xy[first],
        walls=walls,
        grid_px=grid_px,
        clearance_px=clearance_px,
    )
    paths.append(path0)
    # 把路径采样点加入树节点
    for p in _sample_nodes_on_polyline(path0, grid_px):
        if p not in tree_nodes:
            tree_nodes.append(p)
    joins.append((0, first + 1, path0[-1]))  # 0=Panel, first+1=JB
    remaining.remove(first)

    # 按“到树的最短路径”逐个接入
    while remaining:
        best = None
        best_j = None
        best_to = None
        best_path = None

        # 先选若干最近的候选树节点（按曼哈顿）
        if len(tree_nodes) > k_attach:
            # 取所有 JB 的最近树节点集合
            # 为控制复杂度，我们对每个 JB 先取 k_attach 个最近树节点
            for j in remaining:
                tn_sorted = sorted(tree_nodes, key=lambda p: manhattan(jb_xy[j], p))[
                    :k_attach
                ]
                for t in tn_sorted:
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
        else:
            # 树很小：尝试所有树节点
            for j in remaining:
                for t in tree_nodes:
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

        if best_path is None:
            # 兜底：直接连 panel（避免卡住）
            j = remaining.pop()
            path = route_rect_with_walls(
                panel_coord,
                jb_xy[j],
                walls=walls,
                grid_px=grid_px,
                clearance_px=clearance_px,
                node_limit=node_limit,
                deadline_s=deadline_s,
            )
            paths.append(path)
            for p in _sample_nodes_on_polyline(path, grid_px):
                if p not in tree_nodes:
                    tree_nodes.append(p)
            joins.append((0, j + 1, path[-1]))
            continue

        # 成功找到：加入树
        paths.append(best_path)
        for p in _sample_nodes_on_polyline(best_path, grid_px):
            if p not in tree_nodes:
                tree_nodes.append(p)
        # 找到 best_to 对应的“树端点”是谁（Panel=0 或 某个 JB）
        # 近似：若 best_to 恰为某 JB 中心则取该 JB，否则认为是树中间点（虚拟接点）
        attach_idx = 0 if best_to == snap(panel_coord, grid_px) else None
        for j in comp:
            if best_to == snap(jb_xy[j], grid_px):
                attach_idx = j + 1
                break
        joins.append(
            (attach_idx if attach_idx is not None else -1, best_j + 1, best_to)
        )
        remaining.remove(best_j)

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


# ==================== 双 Panel（加入史坦纳主干） ====================


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
    mark_virtual_steiner: bool = True,
):
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

    # 设备点
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

    # 设备→JB
    local2jb = {}
    for gidx, jb in dev2jb.items():
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
        lab_i = dets_side[kept_idx[local_i]]["label"]
        col_i = color_map.get(lab_i) if color_map else color
        draw_polyline_annot(
            page,
            poly,
            width=wire_width,
            stroke_color=col_i,
            title="WIRE",
            contents=f"Dev{local_i}({lab_i})->{jb_prefix}{jb_idx}",
        )

    # ======= 史坦纳主干：分量内最短路生长，去重绘制 =======
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

    drawn_segments: set = set()
    virtual_id = 0

    for comp in comps:
        # 为该分量构建主干路径集合
        trunk_paths, joins = build_steiner_trunk_paths(
            panel_coord,
            jb_xy,
            comp,
            walls=walls or [],
            grid_px=grid_px,
            clearance_px=clearance_px,
            k_attach=8,
            node_limit=20000,
            deadline_s=0.30,
        )
        # 绘制路径（只画未绘制过的段）
        for poly, jinfo in zip(trunk_paths, joins):
            u_idx, v_idx, join_point = jinfo
            # 端点名
            u_name = node_with_types_idx(u_idx) if u_idx >= 0 else "TRUNK"
            v_name = node_with_types_idx(v_idx)
            L_pts = rect_path_length(poly)
            L_str = format_length_ft_in(L_pts, inch_precision=0)
            content = f"{u_name} -> {v_name} | L={L_str}"
            # content = f"{L_str}"

            draw_unique_polyline(
                page,
                poly,
                stroke_color=color,
                width=5,
                title="main bus",
                contents=content,
                drawn_segments=drawn_segments,
            )
            # 在“树中间接入”的情形可选打一个虚拟 JB 标记
            if mark_virtual_steiner and u_idx < 0:
                virtual_id += 1
                draw_circle_annot(
                    page,
                    join_point,
                    r=jb_radius * 0.6,
                    fill_color=None,
                    width=0.9,
                    title="JB",
                    contents=f"{jb_prefix}S{virtual_id}",
                )

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
        "trunk_edges": len(drawn_segments),  # 实际绘制的唯一段数
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
        priority_families=None,
        on_line_tol_px=2.0,
    )
    print("Dual-panels routing summary:", info)
