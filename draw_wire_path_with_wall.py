# -*- coding: utf-8 -*-
from __future__ import annotations
import math
from collections import defaultdict
from statistics import median
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import fitz  # PyMuPDF
from heapq import heappush, heappop

Point = Tuple[float, float]
Edge = Tuple[int, int]

# ===== 配色（Okabe–Ito，舒服&色盲友好） =====
OKABE_ITO = [
    (0.0, 0.447, 0.698),  # 蓝
    (0.902, 0.624, 0.000),  # 橙
    (0.000, 0.619, 0.451),  # 绿
    (0.800, 0.475, 0.655),  # 紫
    (0.941, 0.894, 0.259),  # 黄
    (0.835, 0.369, 0.000),  # 朱红
    (0.000, 0.000, 0.000),  # 黑
    (0.600, 0.600, 0.600),  # 灰
]


def pick_device_colors(labels):
    """给设备标签选择颜色：稳定排序后按调色板分配，>8 时循环。"""
    uniq = sorted(set(labels))
    colors = {}
    for i, lab in enumerate(uniq):
        colors[lab] = OKABE_ITO[i % len(OKABE_ITO)]
    return colors


# ==================== 基础度量/工具 ====================


def manhattan(a: Point, b: Point) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def rectilinear_mst(points: List[Point]) -> List[Edge]:
    """基于曼哈顿距离的 Prim 最小生成树（返回边对的索引）。"""
    n = len(points)
    if n <= 1:
        return []
    in_tree = [False] * n
    dist = [float("inf")] * n
    parent = [-1] * n
    dist[0] = 0.0
    edges: List[Edge] = []
    for _ in range(n):
        u = -1
        best = float("inf")
        for i in range(n):
            if not in_tree[i] and dist[i] < best:
                u, best = i, dist[i]
        if u == -1:
            break
        in_tree[u] = True
        if parent[u] != -1:
            edges.append((parent[u], u))
        for v in range(n):
            if not in_tree[v]:
                cost = manhattan(points[u], points[v])
                if cost < dist[v]:
                    dist[v] = cost
                    parent[v] = u
    return edges


def orthogonal_dogleg(a: Point, b: Point, mode: str = "auto") -> List[Point]:
    """a->b 的正交折线（一个拐点）。"""
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
    """L1（曼哈顿）几何中位点：对 x、y 分别取中位数。"""
    ps = list(points)
    xs = [p[0] for p in ps]
    ys = [p[1] for p in ps]
    return (median(xs), median(ys))


def snap(p: Point, step: float) -> Point:
    if step and step > 0:
        return (round(p[0] / step) * step, round(p[1] / step) * step)
    return p


# ==================== 线段几何（避墙用） ====================


def clamp01(t):
    return 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)


def seg_seg_intersect(p0, p1, q0, q1, *, inclusive=True) -> bool:
    """线段相交（含端点）。只要有真正几何相交就返回 True。"""
    (x1, y1), (x2, y2) = p0, p1
    (x3, y3), (x4, y4) = q0, q1

    def orient(ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    o1 = orient(x1, y1, x2, y2, x3, y3)
    o2 = orient(x1, y1, x2, y2, x4, y4)
    o3 = orient(x3, y3, x4, y4, x1, y1)
    o4 = orient(x3, y3, x4, y4, x2, y2)

    if inclusive:
        # 一般相交 or 共线且投影重叠
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
    """两线段最小距离（欧式）。"""
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
    d = min(
        point_seg_dist(x1, y1, x3, y3, x4, y4),
        point_seg_dist(x2, y2, x3, y3, x4, y4),
        point_seg_dist(x3, y3, x1, y1, x2, y2),
        point_seg_dist(x4, y4, x1, y1, x2, y2),
    )
    return d


def segment_hits_walls(
    seg_a: Point, seg_b: Point, walls: List[List[Point]], *, clearance_px: float
) -> bool:
    """判断一条（水平/垂直）段是否与任何墙段相交或距离小于净距。"""
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
    """去除共线冗余点；保证只保留拐点。"""
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


# ======== A* 网格路由（正交） ========


def a_star_rectilinear(
    start: Point,
    goal: Point,
    *,
    grid_px: float,
    walls: List[List[Point]],
    clearance_px: float,
    bbox_margin_cells: int = 6,
) -> Optional[List[Point]]:
    """
    在一个以 start/goal 包围盒为主的有限网格上做 A*，四邻接，代价=曼哈顿步长。
    仅在“网格边不碰墙”的情况下允许通过。
    """
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

    return None  # 搜不到


def route_rect_with_walls(
    a: Point, b: Point, *, walls: List[List[Point]], grid_px: float, clearance_px: float
) -> List[Point]:
    """
    先试最短的 L 形折线；如果碰墙，再用 A* 网格路由；最终返回正交折线点列。
    """
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
    if path:
        return path

    # 兜底（允许贴墙但不穿越的 L 形）
    return cand


# ==================== 从 PDF 提取设备点 ====================


def get_word_centers(
    page: fitz.Page, target_text: str, case_sensitive: bool = True
) -> List[Point]:
    words = page.get_text("words")  # (x0, y0, x1, y1, w, block, line, word_no)
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
    """一次把多个标签的设备都找出来，返回统一的 dets 列表。"""
    dets_all: List[Dict] = []
    for lab in device_labels:
        pts = get_word_centers(page, lab, case_sensitive=True)
        for x, y in pts:
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


# ==================== JB 自动放置（基于全量设备） ====================


def auto_place_junction_boxes(
    dets: List[Dict],
    *,
    capacity: int = 8,
    eps_px: float = 400,
    grid_px: float = 50,
    merge_eps: float = 50,
    min_conf: float = 0.30,
    keep_labels: Optional[Iterable[str]] = None,  # None=除了 PANEL 全保留
    return_coords: bool = False,
):
    """
    dets: 每个元素形如 {"x1","y1","x2","y2","label","confidence"}
    返回:
        jb_centroids  : [(x_px, y_px), ...]
        dev2jb        : {device_global_idx -> jb_idx}
        groups        : {jb_idx -> [device_global_idx]}
        dev_coords(*) : 仅 return_coords=True 才返回；为“被保留设备”的中心点（与 kept_idx 对齐）
    """
    # 0) 设备筛选
    if keep_labels is None:
        all_labels = {d["label"] for d in dets if d.get("label") is not None}
        keep_labels = {l for l in all_labels if l != "PANEL"}
    keep_labels = set(keep_labels)

    centres = []
    kept_idx = []
    for i, d in enumerate(dets):
        if d.get("confidence", 1.0) < min_conf:
            continue
        if d.get("label") not in keep_labels:
            continue
        cx = (d["x1"] + d["x2"]) * 0.5
        cy = (d["y1"] + d["y2"]) * 0.5
        centres.append([cx, cy])
        kept_idx.append(i)
    centres = np.asarray(centres, float)
    if centres.size == 0:
        raise ValueError("No valid devices after filtering")

    # 1) 合并并排插座
    merge_lbl = DBSCAN(eps=merge_eps, min_samples=1).fit_predict(centres)
    merged = []
    orig2merged_local = {}  # kept_idx 的局部索引 -> merged 索引
    for g in np.unique(merge_lbl):
        idx = np.where(merge_lbl == g)[0]
        pt = centres[idx].mean(axis=0)
        m_idx = len(merged)
        merged.append(pt)
        for i_local in idx:
            orig2merged_local[i_local] = m_idx
    merged = np.asarray(merged, float)

    # 2) JB 聚类 + 容量切分
    big_lbl = DBSCAN(eps=eps_px, min_samples=1).fit_predict(merged)
    final_lbl = -np.ones(len(merged), int)
    jb_xy: List[Point] = []
    cid = 0
    for c in np.unique(big_lbl):
        idx = np.where(big_lbl == c)[0]
        need = int(np.ceil(len(idx) / float(capacity)))
        if need <= 1:
            sub = np.zeros(len(idx), int)
        else:
            sub = KMeans(n_clusters=need, n_init="auto").fit_predict(merged[idx])
        for s in np.unique(sub):
            sub_idx = idx[sub == s]
            pt = merged[sub_idx].mean(axis=0)
            pt = snap((pt[0], pt[1]), grid_px)
            jb_xy.append(pt)
            final_lbl[sub_idx] = cid
            cid += 1

    # 3) 映射回 dets 的全局索引
    dev2jb_local = {}
    for orig_local, m_i in orig2merged_local.items():
        dev2jb_local[orig_local] = int(final_lbl[m_i])

    dev2jb: Dict[int, int] = {}
    for local_i, jb in dev2jb_local.items():
        dev2jb[kept_idx[local_i]] = jb

    groups = defaultdict(list)
    for d_idx, jb in dev2jb.items():
        groups[jb].append(d_idx)

    if return_coords:
        dev_coords = [tuple(centres[i_local]) for i_local in range(len(centres))]
        return jb_xy, dev2jb, dict(groups), dev_coords
    return jb_xy, dev2jb, dict(groups)


# ==================== 注释绘制（红色，Multiply 混合） ====================


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


# 比例：1/8" = 1'-0"
SCALE_IN_PER_FT = 1.0 / 8.0  # 每 1 ft 对应纸面 1/8 英寸
PTS_PER_INCH = 72.0
PTS_PER_FT = PTS_PER_INCH * SCALE_IN_PER_FT  # = 9 pt/ft


def rect_path_length(pts):
    """正交折线总曼哈顿长度（单位：pt）"""
    if not pts or len(pts) < 2:
        return 0.0
    total = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        total += abs(x1 - x0) + abs(y1 - y0)
    return total


def format_length_ft_in(
    L_pts: float, *, pts_per_ft: float = PTS_PER_FT, inch_precision: int = 0
) -> str:
    """
    把长度(单位 pt)格式化为 英尺-英寸 文字，例如 27'-3"
    inch_precision: 英寸小数位（0=取整英寸；1=到0.1"；2=到0.01"）
    """
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
    if inch_precision == 0:
        inches_str = f'{int(inches)}"'
    else:
        inches_str = f'{inches:.{inch_precision}f}"'
    return f"{whole_ft}'-{inches_str}"


# ==================== 放置 JB + 线路整合（支线+主干，支持避墙/彩色） ====================


def route_with_jb_strategy(
    input_pdf: str,
    output_pdf: str,
    page_index: int,
    dets: List[Dict],
    panel_coord: Point,
    color,  # 默认色（主干或兜底）
    *,
    capacity=8,
    eps_px=400,
    grid_px=50,
    merge_eps=50,
    min_conf=0.30,
    keep_labels=None,
    jb_radius=10.0,
    dev_radius=6.0,
    panel_radius=12.0,
    wire_width=1.0,
    walls: Optional[List[List[Point]]] = None,
    clearance_px: float = 0.0,
    color_map: Optional[Dict[str, Tuple[float, float, float]]] = None,  # 按标签着色
):
    """
    策略：
      1) 用 auto_place_junction_boxes 放置 JB（红色注释显示）
      2) 每个设备 -> 其 JB 画一条正交折线（支线），支持避墙
      3) 面板 + 所有 JB 用 MST 连起来（主干），支持避墙
    """
    doc = fitz.open(input_pdf)
    out = fitz.open()
    out.insert_pdf(doc)
    doc.close()
    page = out[page_index]

    jb_xy, dev2jb, groups, dev_coords = auto_place_junction_boxes(
        dets,
        capacity=capacity,
        eps_px=eps_px,
        grid_px=grid_px,
        merge_eps=merge_eps,
        min_conf=min_conf,
        keep_labels=keep_labels,
        return_coords=True,
    )

    # 画 JB（空心红圈）
    for j, pt in enumerate(jb_xy):
        draw_circle_annot(
            page,
            pt,
            r=jb_radius,
            fill_color=None,
            width=1.1,
            title="JB",
            contents=f"JB{j}",
        )

    # 构造 kept_idx（被保留的 det 索引），与 dev_coords 对齐
    kept_idx = [
        i
        for i, d in enumerate(dets)
        if d.get("confidence", 1.0) >= min_conf
        and d.get("label")
        in (keep_labels or {d["label"] for d in dets if d.get("label") != "PANEL"})
    ]

    # 画设备（实心点，按标签着色）
    kept_centres = dev_coords
    for local_i, (x, y) in enumerate(kept_centres):
        lab_i = dets[kept_idx[local_i]]["label"]
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

    # 支线：设备 -> JB（按设备标签颜色）
    # 先把 dev2jb 的“全局索引”映回“保留序号”
    local2jb = {}
    for gidx, jb in dev2jb.items():
        # gidx 一定在 kept_idx 内
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
        lab_i = dets[kept_idx[local_i]]["label"]
        col_i = color_map.get(lab_i) if color_map else color
        draw_polyline_annot(
            page,
            poly,
            width=wire_width,
            stroke_color=col_i,
            title="WIRE",
            contents=f"Dev{local_i}({lab_i})->JB{jb_idx}",
        )

    # 主干：面板 + 所有 JB 的 MST（主干颜色使用传入的 color）
    trunk_points = [panel_coord] + jb_xy
    trunk_edges = rectilinear_mst(trunk_points)

    for u, v in trunk_edges:
        a = trunk_points[u]
        b = trunk_points[v]
        poly = route_rect_with_walls(
            a, b, walls=walls or [], grid_px=grid_px, clearance_px=clearance_px
        )
        L_pts = rect_path_length(poly)
        L_str = format_length_ft_in(L_pts, inch_precision=0)

        draw_polyline_annot(
            page,
            poly,
            width=wire_width,
            stroke_color=color,
            title="main bus",
            contents=f"T{u}->{v} | L={L_str}",
        )

    # 面板：大实心点
    draw_circle_annot(
        page,
        panel_coord,
        r=panel_radius,
        fill_color=color,
        width=0.4,
        title="PANEL",
    )

    out.save(output_pdf)
    out.close()

    return {
        "num_jb": len(jb_xy),
        "dev_count": len(kept_centres),
        "groups": {k: v for k, v in groups.items()},
        "trunk_edges": len(trunk_edges),
    }


# ==================== 示例：一次性提取所有设备并路由 ====================

if __name__ == "__main__":
    # === 使用示例（按需修改） ===
    file_name = "pdf_files/L0_A"
    input_pdf = f"{file_name}.pdf"
    output_pdf_route = f"{file_name}_route_with_JB.pdf"

    PAGE_INDEX = 0
    DEVICE_LABEL = ["2", "4", "6", "49", "18"]
    color_map = pick_device_colors(DEVICE_LABEL)
    PANEL_COORD: Point = (1428.0, 1032.0)

    WALLS = [
        [
            (1384.6190185546875, 1133.4849853515625),
            (1384.6190185546875, 1148.760009765625),
            (1551.1920166015625, 1148.760009765625),
            (1551.1920166015625, 884.35302734375),
            (1383.8919677734375, 884.35302734375),
            (1383.8919677734375, 1076.02099609375),
        ],
        [
            (1124.5830078125, 977.93994140625),
            (1207.9420166015625, 977.93994140625),
            (1207.9420166015625, 1101.886962890625),
            (1306.5760498046875, 1101.886962890625),
            (1306.5760498046875, 975.758056640625),
            (1267.2969970703125, 975.758056640625),
        ],
        [
            (943.1375122070312, 2663.236328125),
            (676.47509765625, 2663.236328125),
            (676.47509765625, 2260.673828125),
            (1024.532958984375, 2260.673828125),
            (1024.532958984375, 2662.631103515625),
            (976.08837890625, 2662.631103515625),
        ],
        [
            (1551.27294921875, 883.330078125),
            (1551.27294921875, 662.868896484375),
            (1252.883056640625, 662.868896484375),
            (1252.883056640625, 885.097900390625),
            (1308.1319580078125, 885.097900390625),
        ],
        [
            (1253.02294921875, 662.490966796875),
            (1163.1090087890625, 662.490966796875),
            (1163.1090087890625, 885.508056640625),
            (1194.9329833984375, 885.508056640625),
        ],
        [
            (1163.4649658203125, 661.281982421875),
            (915.22021484375, 661.281982421875),
            (915.22021484375, 888.576904296875),
            (1118.947998046875, 888.576904296875),
        ],
    ]
    CLEARANCE = 3.0  # pt，按需调整

    # === 一次性提取“所有设备” ===
    doc = fitz.open(input_pdf)
    page = doc[PAGE_INDEX]
    dets_all = build_dets_from_pdf_text_multi(page, DEVICE_LABEL)
    doc.close()

    # === 一次性放置 JB + 画所有支线和主干 ===
    info = route_with_jb_strategy(
        input_pdf=input_pdf,
        output_pdf=output_pdf_route,
        page_index=PAGE_INDEX,
        dets=dets_all,  # <--- 传入全量设备
        panel_coord=PANEL_COORD,
        color=(1, 0, 0),  # 默认单色（主干/兜底使用）
        capacity=8,
        eps_px=400,
        grid_px=50,
        merge_eps=50,
        min_conf=0.30,
        keep_labels=set(DEVICE_LABEL),  # 只保留这些标签
        jb_radius=10.0,
        dev_radius=6.0,
        panel_radius=12.0,
        wire_width=3.0,
        walls=WALLS,
        clearance_px=CLEARANCE,
        color_map=color_map,  # 按标签上色
    )
    print("Route with JB (all devices at once):", info)
