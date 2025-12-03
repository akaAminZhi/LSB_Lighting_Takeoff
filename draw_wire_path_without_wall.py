# -*- coding: utf-8 -*-
from __future__ import annotations
import math
from collections import defaultdict
from statistics import median
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import fitz  # PyMuPDF

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


# ==================== 常用工具 ====================


def pick_device_colors(labels):
    """给设备标签选择颜色：稳定排序后按调色板分配，>8 时循环。"""
    uniq = sorted(set(labels))
    colors = {}
    for i, lab in enumerate(uniq):
        colors[lab] = OKABE_ITO[i % len(OKABE_ITO)]
    return colors


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


# ==================== 从 PDF 提取设备点（可改你自己的） ====================


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


# ==================== 你的 JB 自动放置算法（PDF 版） ====================


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
    dets: 每个元素形如 {"x1","y1","x2","y2","label","confidence"}  (像你贴的那种)
    返回:
        jb_centroids  : [(x_px, y_px), ...]         # 每个 JB 的坐标（已吸附到网格）
        dev2jb        : {device_idx -> jb_idx}      # 设备 → JB 的映射（按 dets 过滤后的顺序）
        groups        : {jb_idx -> [device_idx]}    # JB → 设备列表（索引同上）
        dev_coords(*) : [(x_px,y_px), ...]          # return_coords=True 才返回
    """
    # 0) 设备筛选
    if keep_labels is None:
        all_labels = {d["label"] for d in dets if d.get("label") is not None}
        keep_labels = {l for l in all_labels if l != "PANEL"}
    keep_labels = set(keep_labels)

    centres = []
    kept_idx = []  # 记录 dets 里保留的设备索引（用于最终映射）
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

    # 1) 合并并排插座（近邻合并）
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

    # 3) 映射回原始（保留后的）设备索引空间
    dev2jb_local = {}
    for orig_local, m_i in orig2merged_local.items():
        dev2jb_local[orig_local] = int(final_lbl[m_i])

    # 4) 再映射回 dets 的全局索引（给调用者更直观）
    dev2jb: Dict[int, int] = {}
    for local_i, jb in dev2jb_local.items():
        dev2jb[kept_idx[local_i]] = jb

    groups = defaultdict(list)
    for d_idx, jb in dev2jb.items():
        groups[jb].append(d_idx)

    if return_coords:
        # 返回 dev_coords（按 dets 的下标排序的已过滤坐标）
        dev_coords = [tuple(centres[i_local]) for i_local in range(len(centres))]
        return jb_xy, dev2jb, dict(groups), dev_coords
    return jb_xy, dev2jb, dict(groups)


# ==================== 注释绘制（全红） ====================


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
    # 四舍五入到指定精度
    factor = 10**inch_precision
    inches = round(inches * factor) / factor

    # 处理 12.0" 进位
    if inches >= 12.0:
        whole_ft += 1
        inches -= 12.0

    if inch_precision == 0:
        inches_str = f'{int(inches)}"'
    else:
        inches_str = f'{inches:.{inch_precision}f}"'
    return f"{whole_ft}'-{inches_str}"


# ==================== 放置 JB + 线路整合（支线+主干） ====================


def route_with_jb_strategy(
    input_pdf: str,
    output_pdf: str,
    page_index: int,
    dets: List[Dict],
    panel_coord: Point,
    color,
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
):
    """
    策略：
      1) 用 auto_place_junction_boxes 放置 JB（红色注释显示）
      2) 每个设备 -> 其 JB 画一条正交折线（支线）
      3) 面板 + 所有 JB 用 MST 连起来（主干）
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
        # page.add_text_annot(
        #     fitz.Point(pt[0] + jb_radius + 3, pt[1] - jb_radius - 3), f"JB{j}"
        # ).update()

    # 画设备（实心红点）
    kept_centres = dev_coords  # 与 dev2jb 的 key 对齐（是筛选后顺序）
    for idx, (x, y) in enumerate(kept_centres):
        draw_circle_annot(
            page,
            (x, y),
            r=dev_radius,
            fill_color=color,
            width=0.2,
            title="DEVICE",
            contents=f"Dev{idx}",
        )

    # 支线：设备 -> JB
    # 注意：dev2jb 的 key 是“dets里被保留设备”的全局索引；我们需要一个“保留序号 -> 全局索引”的映射
    # 便于展示这里简化：我们根据 dev_coords 的顺序（保留序）来画；因此先构造 local->jb
    # 构造 kept_idx（被保留的 det 索引）
    kept_idx = [
        i
        for i, d in enumerate(dets)
        if d.get("confidence", 1.0) >= min_conf
        and d.get("label")
        in (keep_labels or {d["label"] for d in dets if d.get("label") != "PANEL"})
    ]
    local2jb = {}
    for gidx, jb in dev2jb.items():
        local = kept_idx.index(gidx)  # 把全局索引映射回“保留序号”
        local2jb[local] = jb

    for local_i, (x, y) in enumerate(kept_centres):
        if local_i not in local2jb:
            continue
        jb_idx = local2jb[local_i]
        jx, jy = jb_xy[jb_idx]
        poly = orthogonal_dogleg((x, y), (jx, jy), mode="auto")
        draw_polyline_annot(
            page,
            poly,
            width=wire_width,
            stroke_color=color,
            title="WIRE",
            contents=f"Dev{local_i}->JB{jb_idx}",
        )

    # 主干：面板 + 所有 JB 的 MST
    trunk_points = [panel_coord] + jb_xy
    trunk_edges = rectilinear_mst(trunk_points)

    for u, v in trunk_edges:
        a = trunk_points[u]
        b = trunk_points[v]
        poly = orthogonal_dogleg(a, b, mode="auto")
        L_pts = rect_path_length(poly)
        L_str = format_length_ft_in(L_pts, inch_precision=0)  # 例如 27'-3"

        draw_polyline_annot(
            page,
            poly,
            width=wire_width,
            stroke_color=color,
            title="main bus",
            contents=f"T{u}->{v} | L={L_str}",
        )
    # 面板：大实心红点
    draw_circle_annot(
        page,
        panel_coord,
        r=panel_radius,
        fill_color=color,
        width=0.4,
        title="PANEL",
    )
    # page.add_text_annot(
    #     fitz.Point(
    #         panel_coord[0] + panel_radius + 4, panel_coord[1] - panel_radius - 4
    #     ),
    #     "PANEL",
    # ).update()

    out.save(output_pdf)
    out.close()
    # doc.close()

    return {
        "num_jb": len(jb_xy),
        "dev_count": len(kept_centres),
        "groups": {k: v for k, v in groups.items()},
        "trunk_edges": len(trunk_edges),
    }


# ==================== 示例：把 PDF 上的 "49" 当作设备 ====================


def build_dets_from_pdf_text(page: fitz.Page, device_label: str) -> List[Dict]:
    """把 PDF 上匹配到的文字中心点转成 dets 结构（confidence=1.0）。"""
    pts = get_word_centers(page, device_label, case_sensitive=True)
    dets = []
    for x, y in pts:
        # 给个小 bbox（不重要，仅用于一致的字段）
        dets.append(
            {
                "x1": x - 2,
                "y1": y - 2,
                "x2": x + 2,
                "y2": y + 2,
                "label": device_label,
                "confidence": 1.0,
            }
        )
    return dets


if __name__ == "__main__":
    # === 使用示例（按需修改） ===
    file_name = "pdf_files/L0_A"
    input_pdf = f"{file_name}.pdf"
    output_pdf_jb = f"{file_name}_JB_only.pdf"
    output_pdf_route = f"{file_name}_route_with_JB.pdf"

    PAGE_INDEX = 0
    DEVICE_LABEL = ["2", "4", "6", "49"]
    color_map = pick_device_colors(DEVICE_LABEL)
    PANEL_COORD: Point = (1428.0, 1032.0)

    for device in DEVICE_LABEL:
        # 1) 仅放置并绘制 JB（红色注释）
        doc = fitz.open(input_pdf)
        dets = build_dets_from_pdf_text(doc[PAGE_INDEX], device)
        doc.close()

        # 2) 放置 JB + 画支线和主干（都为红色注释）
        info2 = route_with_jb_strategy(
            input_pdf=input_pdf,
            output_pdf=output_pdf_route,
            page_index=PAGE_INDEX,
            dets=dets,
            panel_coord=PANEL_COORD,
            color=color_map[device],
            capacity=8,
            eps_px=400,
            grid_px=50,
            merge_eps=50,
            min_conf=0.30,
            keep_labels=None,
            jb_radius=10.0,
            dev_radius=6.0,
            panel_radius=12.0,
            wire_width=3.0,
        )
        print("Route with JB:", info2)
        input_pdf = output_pdf_route
