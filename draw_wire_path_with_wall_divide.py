# -*- coding: utf-8 -*-
from __future__ import annotations
import math
from collections import defaultdict, Counter
from statistics import median
from typing import Dict, List, Tuple, Iterable, Optional, Set

import numpy as np
from sklearn.cluster import DBSCAN
import fitz  # PyMuPDF
from heapq import heappush, heappop

Point = Tuple[float, float]
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


def seg_seg_distance(p0, p1, q0, q1) -> float:
    if seg_seg_intersect(p0, p1, q0, q1):
        return 0.0

    def point_seg_dist(px, py, ax, ay, bx, by):
        vx, vy = bx - ax, by - ay
        ux, uy = px - ax, py - ay
        denom = vx * vx + vy * vy
        t = 0.0 if denom == 0 else clamp01((ux * vx + uy * vy) / denom)
        cx, cy = ax + t * vx, ay + t * vy
        dx, dy = px - cx, py - cy
        return math.hypot(dx, dy)

    (x1, y1), (x2, y2) = p0, p1
    (x3, y3), (x4, y4) = q0, q1
    return min(
        point_seg_dist(x1, y1, x3, y3, x4, y4),
        point_seg_dist(x2, y2, x3, y3, x4, y4),
        point_seg_dist(x3, y3, x1, y1, x2, y2),
        point_seg_dist(x4, y4, x1, y1, x2, y2),
    )


def segment_hits_walls(
    seg_a: Point, seg_b: Point, walls: List[List[Point]], *, clearance_px: float
) -> bool:
    if not walls:
        return False
    for poly in walls:
        if len(poly) < 2:
            continue
        for u, v in zip(poly, poly[1:]):
            if clearance_px <= 1e-6:
                if seg_seg_intersect(seg_a, seg_b, u, v, inclusive=True):
                    return True
            else:
                if seg_seg_distance(seg_a, seg_b, u, v) < clearance_px - 1e-9:
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


# ==================== A* 网格路由（避墙） ====================


def a_star_rectilinear(
    start: Point,
    goal: Point,
    *,
    grid_px: float,
    walls: List[List[Point]],
    clearance_px: float,
    bbox_margin_cells: int = 6,
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

    openq = []
    heappush(openq, (h(s), 0.0, s))
    came = {}
    gscore = {s: 0.0}
    visited = set()
    while openq:
        _, gs, cur = heappop(openq)
        if cur in visited:
            continue
        visited.add(cur)
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
    a: Point, b: Point, *, walls: List[List[Point]], grid_px: float, clearance_px: float
) -> List[Point]:
    cand = orthogonal_dogleg(a, b, mode="auto")
    ok = True
    for u, v in zip(cand, cand[1:]):
        if segment_hits_walls(u, v, walls, clearance_px=clearance_px):
            ok = False
            break
    if ok:
        return cand
    path = a_star_rectilinear(
        a, b, grid_px=grid_px, walls=walls, clearance_px=clearance_px
    )
    return path if path else cand


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
    dets_all: List[Dict] = []
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


# ==================== 全局不相交家族（≤3） ====================


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

    # 1) 全量共现计数（基于空间邻近）
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

    # 2) 并查集初始化
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
        # 家族最多 3 种
        if size[rx] + size[ry] > max_family_size:
            return rx  # 拒绝
        parent[ry] = rx
        size[rx] += size[ry]
        return rx

    # 2a) 先落地优先家族（硬约束）
    if priority_families:
        for fam in priority_families:
            fam = sorted(set(fam) & set(labels))
            if not fam:
                continue
            base = idx_of_label[fam[0]]
            for lab in fam[1:]:
                base = union(base, idx_of_label[lab])

    # 3) 按共现从大到小合并（不跨 3 种上限）
    for (a, b), _w in co.most_common():
        if a not in idx_of_label or b not in idx_of_label:
            continue
        union(idx_of_label[a], idx_of_label[b])

    # 4) 导出 families
    fam_id_map: Dict[int, int] = {}
    families: List[Set[str]] = []
    for lab, i in idx_of_label.items():
        r = find(i)
        if r not in fam_id_map:
            fam_id_map[r] = len(families)
            families.append(set())
        families[fam_id_map[r]].add(lab)
    family_id_of_label = {lab: fam_id_map[find(idx_of_label[lab])] for lab in labels}
    return family_id_of_label, families


# ==================== JB 放置（按标签预合并 + 家族硬约束） ====================


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
):
    """
    关键：并排合并改为“**按标签分组**后再 DBSCAN”，确保原子设备点是**单标签**。
    JB 合并与 JB↔JB 主干连接都在**同一家族**内执行，且标签并集 ≤ 3。
    """
    if keep_labels is None:
        all_labels = {d["label"] for d in dets if d.get("label") is not None}
        keep_labels = {l for l in all_labels if l != "PANEL"}
    keep_labels = set(keep_labels)

    # 0) 全局家族（不相交，≤3）
    family_id_of_label, families = build_disjoint_families(
        dets,
        keep_labels,
        neighbor_eps_px=eps_px,
        min_conf=min_conf,
        priority_families=priority_families,
        max_family_size=3,
    )

    # 1) 过滤（保留设备）
    raw_pts: List[Tuple[float, float]] = []
    raw_idx: List[int] = []
    raw_labs: List[str] = []
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

    # 2) 并排预合并（**逐标签** DBSCAN）：原子设备点必为**单标签**
    dev_points: List[Point] = []
    dev_label_sets: List[Set[str]] = []
    dev_counts: List[int] = []
    dev_point_to_global_idxs: List[List[int]] = []

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
            members = [idxs[s] for s in sub]
            p = pts_lab[sub].mean(axis=0)
            dev_points.append((float(p[0]), float(p[1])))
            dev_label_sets.append({lab})  # 单标签
            dev_counts.append(len(members))
            dev_point_to_global_idxs.append([raw_idx[m] for m in members])

    dev_points_np = np.asarray(dev_points, float)

    # 3) 初始 JB = 原子设备点（1:1）
    JB_centers = [snap(tuple(p), grid_px) for p in dev_points_np]
    JB_label_sets = [set(s) for s in dev_label_sets]  # 单标签
    JB_counts = list(dev_counts)
    JB_members: List[List[int]] = [[i] for i in range(len(dev_points))]
    active = [True] * len(JB_centers)

    def same_family(L: Set[str]) -> bool:
        if not L:
            return True
        fids = {family_id_of_label[t] for t in L if t in family_id_of_label}
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
        L_i = JB_label_sets[i]
        L_j = JB_label_sets[j]
        inter = len(L_i & L_j)
        new_added = len((L_i | L_j)) - max(len(L_i), len(L_j))
        dist = manhattan(JB_centers[i], JB_centers[j])
        return (inter, -new_added, -dist)

    # 4) 贪心凝聚（家族硬约束）
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
        JB_centers[i] = snap(new_center, grid_px)
        active[j] = False

    # 5) 压缩有效 JB
    jb_xy: List[Point] = []
    jb_label_sets_out: List[Set[str]] = []
    jb_members: List[List[int]] = []
    for old_i, on in enumerate(active):
        if not on:
            continue
        jb_xy.append(JB_centers[old_i])
        jb_label_sets_out.append(JB_label_sets[old_i])
        jb_members.append(JB_members[old_i])

    # 6) dev2jb & groups
    dev2jb: Dict[int, int] = {}
    groups = defaultdict(list)
    for new_jb_idx, members in enumerate(jb_members):
        for dp in members:
            for g in dev_point_to_global_idxs[dp]:
                dev2jb[g] = new_jb_idx
                groups[new_jb_idx].append(g)

    if return_coords:
        dev_coords = [tuple(p) for p in raw_pts]  # 过滤后的原始中心点（用于绘制设备）
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


def draw_polyline_annot(
    page: fitz.Page,
    pts: List[Point],
    width: float = 0.9,
    stroke_color=(1, 0, 0),
    title: str = "WIRE",
    contents: str = "",
):
    if len(pts) < 2:
        return None
    plist = [fitz.Point(p[0], p[1]) for p in pts]
    annot = page.add_polyline_annot(plist)
    annot.set_colors(stroke=stroke_color)
    annot.set_border(width=width)
    annot.set_info(subject=title)
    annot.set_blendmode("Multiply")
    if title:
        annot.set_info(title=title)
    if contents:
        annot.set_info(content=contents)
    annot.update()
    return annot


# ==================== 长度格式化 ====================

SCALE_IN_PER_FT = 1.0 / 8.0
PTS_PER_INCH = 72.0
PTS_PER_FT = PTS_PER_INCH * SCALE_IN_PER_FT  # = 9 pt/ft


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


# ==================== JB↔JB 受限森林（同家族） + 每分量各自连 Panel ====================


def jb_forest_with_per_component_panel_links(
    panel_coord: Point,
    jb_xy: List[Point],
    jb_label_sets: List[Set[str]],
    *,
    max_types: int = 3,
    family_id_of_label: Dict[str, int],
    families: List[Set[str]],
) -> List[Edge]:
    """
    仅允许同家族的 JB 分量相连；并集 L 要满足：
      - 所有标签 family id 相同；且 L ⊆ families[fid]
      - |L| ≤ max_types
    Panel 不参与并查集；每个分量各自 homerun 到 Panel。
    返回边 index 基于 trunk_points = [panel] + jb_xy（JB 要 +1）
    """
    m = len(jb_xy)
    if m == 0:
        return []

    def same_family(L: Set[str]) -> Optional[int]:
        if not L:
            return None
        fids = {family_id_of_label[t] for t in L if t in family_id_of_label}
        if len(fids) != 1:
            return None
        fid = next(iter(fids))
        return fid if L.issubset(families[fid]) else None

    parent = list(range(m))
    rank = [0] * m
    comp_labels: List[Set[str]] = [set(s) for s in jb_label_sets]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return rx
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        comp_labels[rx] |= comp_labels[ry]
        return rx

    cand = []
    for i in range(m):
        for j in range(i + 1, m):
            cand.append((manhattan(jb_xy[i], jb_xy[j]), i, j))
    cand.sort(key=lambda t: t[0])

    chosen_edges: List[Edge] = []
    for _, i, j in cand:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        U = comp_labels[ri] | comp_labels[rj]
        if len(U) > max_types:
            continue
        fid = same_family(U)
        if fid is None:
            continue
        union(ri, rj)
        chosen_edges.append((i + 1, j + 1))  # +1: panel 在 0

    # 每个分量接 Panel（离 panel 最近的 JB）
    reps: Dict[int, List[int]] = {}
    for v in range(m):
        r = find(v)
        reps.setdefault(r, []).append(v)
    for r, members in reps.items():
        best = min(members, key=lambda k: manhattan(jb_xy[k], panel_coord))
        chosen_edges.append((0, best + 1))
    return chosen_edges


# ==================== 辅助：分割线左右判定 ====================


def _point_side_of_line(
    p: Point, a: Point, b: Point, *, on_line_tol_px: float = 2.0
) -> str:
    """
    返回 'L' / 'R' / 'ON'
    以向量 a->b 的左侧为 'L'，右侧为 'R'；到直线的几何距离 <= on_line_tol_px 视为 'ON'。
    注意：PDF 坐标 y 轴向下，不影响叉积符号使用。
    """
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
    """把 dets 分成左/右两组。在线上的分配给更近的 panel。"""
    a, b = divider_line
    left_dets: List[Dict] = []
    right_dets: List[Dict] = []
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
        else:  # "ON"：分给更近的 panel（用 L1）
            d_left = manhattan((cx, cy), left_panel)
            d_right = manhattan((cx, cy), right_panel)
            if d_left <= d_right:
                left_dets.append(d)
            else:
                right_dets.append(d)
    return left_dets, right_dets


# ==================== 双 Panel 总策略（左右完全隔离） ====================


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
    jb_prefix: str,  # "JBL" 或 "JBR"
):
    if not dets_side:
        # 没有该侧设备也要把 panel 画出来
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
        )
    )

    # 画 JB（分别标注前缀）
    for j, pt in enumerate(jb_xy):
        draw_circle_annot(
            page,
            pt,
            r=jb_radius,
            fill_color=None,
            width=1.1,
            title="JB",
            contents=f"{jb_prefix}{j} | types:{sorted(list(jb_label_sets[j]))}",
        )

    # kept_idx（与 dev_coords 对齐）
    kept_idx = [
        i
        for i, d in enumerate(dets_side)
        if d.get("confidence", 1.0) >= min_conf and d.get("label") in keep_labels
    ]
    kept_centres = dev_coords

    # 画设备（按 label 上色）
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

    # local -> jb
    local2jb: Dict[int, int] = {}
    for gidx, jb in dev2jb.items():
        local = kept_idx.index(gidx)
        local2jb[local] = jb

    # 支线（设备→JB）
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

    # 主干：仅同家族 JB 可相连 + 每分量各自接 Panel（该侧 Panel）
    trunk_points = [panel_coord] + jb_xy
    trunk_edges = jb_forest_with_per_component_panel_links(
        panel_coord,
        jb_xy,
        jb_label_sets,
        max_types=3,
        family_id_of_label=family_id_of_label,
        families=families,
    )

    # 注释：把端点编号转成可读
    # 注释：端点名称 & 类型
    def panel_name() -> str:
        return (
            "PANEL_L"
            if jb_prefix == "JBL"
            else ("PANEL_R" if jb_prefix == "JBR" else "PANEL")
        )

    def pretty_node_name(idx: int) -> str:
        if idx == 0:
            return panel_name()
        return f"{jb_prefix}{idx-1}"

    def node_with_types(idx: int) -> str:
        # 面板没有类型；JB 的类型从 jb_label_sets 读取
        if idx == 0:
            return panel_name()
        j = idx - 1
        types = ",".join(sorted(list(jb_label_sets[j])))
        return f"{jb_prefix}{j}({types})"

    for u, v in trunk_edges:
        a = trunk_points[u]
        b = trunk_points[v]
        poly = route_rect_with_walls(
            a, b, walls=walls or [], grid_px=grid_px, clearance_px=clearance_px
        )
        L_pts = rect_path_length(poly)
        L_str = format_length_ft_in(L_pts, inch_precision=0)

        # 在 main bus 注释里加入 JB 的 device 列表
        name_u = node_with_types(u)
        name_v = node_with_types(v)

        draw_polyline_annot(
            page,
            poly,
            width=5,
            stroke_color=color,
            title="main bus",
            contents=f"{name_u} -> {name_v} | L={L_str}",
        )

    # 画该侧 Panel
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
        "trunk_edges": len(trunk_edges),
    }


def route_with_jb_strategy_dual_panels(
    input_pdf: str,
    output_pdf: str,
    page_index: int,
    dets: List[Dict],
    *,
    left_panel_coord: Point,
    right_panel_coord: Point,
    divider_line: Tuple[Point, Point],  # ((x1,y1),(x2,y2))
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
    """
    把页面按 divider_line 分成左右两半：
      - 左半侧所有设备/JB 只在左侧互连，并各分量回 left_panel_coord；
      - 右半侧同理，回 right_panel_coord；
      - 左右两侧绝不互连。
    """
    doc = fitz.open(input_pdf)
    out = fitz.open()
    out.insert_pdf(doc)
    doc.close()
    page = out[page_index]

    if keep_labels is None:
        all_labels = {d["label"] for d in dets if d.get("label") is not None}
        keep_labels = {l for l in all_labels if l != "PANEL"}
    keep_labels = set(keep_labels)

    # 先把 dets 分成左右两组
    left_dets, right_dets = _split_dets_by_line(
        dets,
        divider_line,
        left_panel_coord,
        right_panel_coord,
        min_conf=min_conf,
        keep_labels=keep_labels,
        on_line_tol_px=on_line_tol_px,
    )

    # 分别绘制左右两侧（完全隔离）
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
    )

    # 存盘
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
    file_name = "pdf_files/L2_A"
    input_pdf = f"{file_name}.pdf"
    output_pdf_route = f"{file_name}_route_dual_panels.pdf"

    PAGE_INDEX = 0
    DEVICE_LABEL = ["14", "18", "10", "27", "12", "20", "24"]
    color_map = pick_device_colors(DEVICE_LABEL)

    # 两个 Panel 的坐标
    LEFT_PANEL: Point = (1941.0, 2416.0)
    RIGHT_PANEL: Point = (1913.0, 919.0)

    # 分割线（例：从 (1100,0) 到 (1100,3000) 的垂直线）
    DIVIDER_LINE = ((964.16, 1648.57), (2098.93, 1948.57))

    WALLS = [
        [
            (1886.137939453125, 2098.58447265625),
            (1673.4949951171875, 2098.58447265625),
            (1673.4949951171875, 2002.552001953125),
            (1569.6240234375, 2002.552001953125),
        ],
        [
            (1673.886962890625, 2097.800537109375),
            (1673.886962890625, 2342.78076171875),
            (1551.9849853515625, 2342.78076171875),
            (1551.9849853515625, 2397.2646484375),
            (1580.20703125, 2397.2646484375),
        ],
        [
            (1490.60205078125, 2776.81689453125),
            (1490.60205078125, 2488.328125),
            (1481.97900390625, 2488.328125),
            (1481.97900390625, 2475.001220703125),
        ],
        [(1490.2099609375, 2384.848388671875), (1490.2099609375, 2000.718994140625)],
        [
            (1673.6199951171875, 2343.513427734375),
            (1673.6199951171875, 2373.455322265625),
            (1617.5469970703125, 2373.455322265625),
            (1617.5469970703125, 2399.586669921875),
        ],
        [
            (1617.8189697265625, 2372.638916015625),
            (1617.8189697265625, 2342.424560546875),
        ],
        [
            (924.5374145507812, 1121.6910400390625),
            (1020.7169799804688, 1121.6910400390625),
            (1020.7169799804688, 1276.1199951171875),
            (1327.81201171875, 1276.1199951171875),
        ],
        [
            (1119.427001953125, 1274.77099609375),
            (1119.427001953125, 1116.72900390625),
            (1017.8280029296875, 1116.72900390625),
        ],
        [
            (1105.6300048828125, 1116.802001953125),
            (1105.6300048828125, 991.68603515625),
            (927.598388671875, 991.68603515625),
            (927.598388671875, 1121.0360107421875),
        ],
        [
            (916.7935180664062, 904.0048828125),
            (916.7935180664062, 694.60009765625),
            (893.0872192382812, 694.60009765625),
            (893.0872192382812, 594.695068359375),
        ],
        [
            (916.2128295898438, 741.239990234375),
            (1018.7780151367188, 741.239990234375),
            (1018.7780151367188, 692.243896484375),
            (1108.2769775390625, 692.243896484375),
            (1108.2769775390625, 586.0859375),
        ],
        [
            (1108.60400390625, 691.263916015625),
            (1108.60400390625, 883.656005859375),
            (1019.4310302734375, 883.656005859375),
        ],
        [(1019.7579956054688, 883.98193359375), (1019.7579956054688, 740.9140625)],
        [
            (1018.4509887695312, 819.634033203125),
            (967.4951782226562, 819.634033203125),
            (967.4951782226562, 879.408935546875),
        ],
        [
            (1888.239013671875, 1370.2239990234375),
            (1962.7969970703125, 1370.2239990234375),
            (1962.7969970703125, 1272.2919921875),
            (1768.4129638671875, 1272.2919921875),
            (1768.4129638671875, 1371.112060546875),
            (1810.7220458984375, 1371.112060546875),
        ],
        [
            (1850.6639404296875, 1371.112060546875),
            (1850.6639404296875, 1271.7010498046875),
        ],
    ]
    CLEARANCE = 9.0

    # 一次性提取全部设备
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
        priority_families=None,  # 如需强制家族，可传如 [{"2","4","6"}]
        on_line_tol_px=2.0,
    )
    print("Dual-panels routing summary:", info)
