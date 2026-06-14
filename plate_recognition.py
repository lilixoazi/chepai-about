"""
基于 HyperLPR 的中国车牌识别脚本（含多层容错增强）
用法: python plate_recognition.py
自动识别 测试照片/ 目录下所有图片，结果输出到终端
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import re
import cv2
import numpy as np
from hyperlpr import HyperLPR_plate_recognition

INPUT_DIR  = "测试照片"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# 中国车牌字符表
PROVINCES = "京津沪渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新"
LETTERS   = "ABCDEFGHJKLMNPQRSTUVWXYZ"
DIGITS    = "0123456789"

_PLATE_RE = re.compile(
    r"^[" + PROVINCES + r"]"
    r"[A-HJ-NP-Z]"
    r"[A-HJ-NP-Z0-9]{5,6}$"
)


# ═══════════════════════════════════════════════════════
# 车牌校验与混淆修正
# ═══════════════════════════════════════════════════════
def is_chinese_plate(s):
    if not s or len(s) < 7:
        return False
    s = s.strip()
    allowed = set(PROVINCES + LETTERS + DIGITS + "港澳学使警挂军")
    return all(c in allowed for c in s) and bool(_PLATE_RE.match(s))


# 视觉混淆邻居表（OCR 易错 → 真实候选，优先级左高右低）
_NEIGHBORS = {
    # 数字 ↔ 相似字母
    '0': ['D', 'Q', 'O'],
    '1': ['L', 'I', 'T', '7'],
    '2': ['Z'],
    '5': ['S'],
    '6': ['G'],
    '7': ['T', '1'],
    '8': ['B'],
    # 字母 ↔ 相似数字/字母
    'B': ['8', 'D'],
    'C': ['G', '6', '0'],
    'D': ['0', 'O', 'Q', 'B'],
    'G': ['6', 'C', '0'],
    'I': ['1', 'L', 'T', '7'],
    'L': ['1', 'I'],
    'O': ['0', 'D', 'Q'],
    'Q': ['0', 'O', 'D'],
    'S': ['5', '6'],
    'T': ['7', '1', 'I'],
    'Z': ['2', '7'],
    # 省份形似
    '京': ['琼', '蒙'],   '琼': ['京'],          '冀': ['翼'],
    '鲁': ['鱼'],         '豫': ['象'],          '湘': ['湖'],
    '蒙': ['家', '京'],   '桂': ['挂'],          '翼': ['冀'],
    '鱼': ['鲁'],         '象': ['豫'],          '湖': ['湘'],
    '家': ['蒙'],         '挂': ['桂'],
}


def _try_fix(chars, idx, validator):
    if validator(chars[idx]):
        return False
    for nb in _NEIGHBORS.get(chars[idx], []):
        if validator(nb):
            chars[idx] = nb
            return True
    return False


def fix_plate(plate_str):
    """逐位修正 OCR 误识（仅修正非法字符），返回 (修正后, 是否修改, 详情)"""
    raw = plate_str.strip()
    if len(raw) < 7:
        return raw, False, []
    chars = list(raw)
    changed, details = False, []

    if _try_fix(chars, 0, lambda c: c in PROVINCES):
        details.append(f"{raw[0]}→{chars[0]}"); changed = True
    if _try_fix(chars, 1, lambda c: c in LETTERS):
        details.append(f"{raw[1]}→{chars[1]}"); changed = True
    for i in range(2, len(chars)):
        if _try_fix(chars, i, lambda c: c in DIGITS or c in LETTERS):
            details.append(f"{raw[i]}→{chars[i]}"); changed = True

    return ''.join(chars), changed, details


# ── 位置先验 ────────────────────────────────────────
# 中国车牌第3~7位：数字远多于字母；末尾几乎全是数字
# Q 等稀有字母在车牌任何位置都很少出现
_RARE_LETTERS  = set("QRSTUVWXYZ")   # 稀有字母
_COMMON_LETTERS = set("ABCDEFGHJKLMNP")  # 常见字母

def _position_prior(plate):
    """
    返回 0.90~1.0 的位置先验因子。
    末尾稀有字母(Q/X/Y/Z等)受轻量惩罚以修正 Q→0 类误识。
    """
    if len(plate) < 7:
        return 1.0
    penalty = 1.0
    body = plate[2:]
    for i, ch in enumerate(body):
        if ch in DIGITS:
            continue
        if ch in LETTERS:
            # 稀有字母(Q/X/Y/Z等)微惩 3%
            if ch in _RARE_LETTERS:
                penalty *= 0.97
            # 末尾位置字母轻微惩罚
            dist_from_end = len(body) - 1 - i
            if dist_from_end == 0:       # 最后一位是字母
                penalty *= 0.92
            elif dist_from_end == 1:     # 倒数第二位是字母
                penalty *= 0.96
    return max(0.90, penalty)


def candidates_from(plate, conf):
    """
    从原始识别结果生成候选列表。
    返回 [(车牌, 置信度, 标签, 详情), ...]
    包含：原始版 / 非法字符修正版 / 混淆邻居试探版
    """
    out = [(plate, conf, "原始", "")]
    seen = {plate}

    # 1. 规则修正（仅非法字符）
    fixed, changed, details = fix_plate(plate)
    if changed and fixed != plate and fixed not in seen:
        seen.add(fixed)
        out.append((fixed, conf * 0.98, "修正", " → ".join(details)))

    # 2. 混淆展开：仅展开可疑字符（稀有字母 + 有视觉邻居的数字）
    #    控制候选数，避免过度修正正常车牌
    _EXPANDABLE = _RARE_LETTERS | set("01568BG")  # 仅易混淆字符参与展开
    chars = list(plate)
    for i in range(len(chars)):
        if chars[i] not in _EXPANDABLE:
            continue
        for nb in _NEIGHBORS.get(chars[i], []):
            alt = chars.copy()
            alt[i] = nb
            alt_str = ''.join(alt)
            if alt_str not in seen and len(alt_str) >= 7 and is_chinese_plate(alt_str):
                seen.add(alt_str)
                # 展开惩罚 0.93 — 仅真正的误识会被位置先验补回
                out.append((alt_str, conf * 0.93, "混淆展开",
                            f"{plate[i]}→{nb}@位{i}"))
                break  # 每位置只取第1个有效邻居

    return out


# ═══════════════════════════════════════════════════════
# I/O（中文路径兼容）
# ═══════════════════════════════════════════════════════
def imread_cn(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# ═══════════════════════════════════════════════════════
# 图像增强函数库
# ═══════════════════════════════════════════════════════
def _sharpen(img, strength=5):
    k = np.array([[0, -1, 0], [-1, strength, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(img, -1, k)


def _clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _gamma(img, g=1.5):
    t = (255.0 * (np.arange(256) / 255.0) ** (1.0 / g)).astype(np.uint8)
    return cv2.LUT(img, t)


def _auto_brightness(img):
    m = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()
    if   m < 80:  return cv2.convertScaleAbs(img, alpha=1.6, beta=30)
    elif m > 170: return cv2.convertScaleAbs(img, alpha=1.2, beta=-20)
    return img


def _denoise(img):
    return cv2.bilateralFilter(img, 9, 75, 75)


def _upscale(img, s=1.5):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)


def _rotate(img, deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _auto_level(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lo, hi = np.percentile(gray, [2, 98])
    if hi - lo < 20: return img
    stretched = np.clip((gray.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)


# ── 车牌专项增强 ──────────────────────────────────────
def _sobel_edge(img):
    """Sobel 边缘增强：强化车牌边框"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge = cv2.convertScaleAbs(cv2.magnitude(sx, sy))
    # 边缘叠加到原图
    edge_bgr = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img, 0.7, edge_bgr, 0.3, 0)


def _morph_close(img):
    """形态学闭运算：填补车牌字符断裂/污损缺口"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def _adaptive_thresh(img):
    """自适应阈值：处理光照不均的牌照表面"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 15, 4)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


def _blue_enhance(img):
    """蓝牌增强：拉升蓝色通道"""
    b, g, r = cv2.split(img)
    b = cv2.equalizeHist(b)
    # 微压红绿通道减少反光干扰
    r = cv2.convertScaleAbs(r, alpha=0.9, beta=0)
    g = cv2.convertScaleAbs(g, alpha=0.9, beta=0)
    return cv2.merge([b, g, r])


def _green_enhance(img):
    """新能源绿牌增强：拉升绿色通道"""
    b, g, r = cv2.split(img)
    g = cv2.equalizeHist(g)
    return cv2.merge([b, g, r])


def _hsv_enhance(img):
    """HSV 空间增强：拉升饱和度 + 明度，应对褪色/老旧牌照"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.convertScaleAbs(s, alpha=1.3, beta=0)
    v = cv2.equalizeHist(v)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


# ═══════════════════════════════════════════════════════
# 增强策略矩阵（图像 × 参数组合）
# ═══════════════════════════════════════════════════════
ENHANCE_STRATEGIES = [
    # (标签, 处理函数)
    ("原图",               lambda img: img),

    # ── 光度 / 对比度 ──
    ("自动亮度",           _auto_brightness),
    ("CLAHE 弱光增强",     _clahe),
    ("Gamma 提亮",         lambda img: _gamma(img, 1.5)),
    ("Gamma 压低高光",     lambda img: _gamma(img, 0.7)),
    ("自动色阶",           _auto_level),

    # ── 模糊 / 分辨率 ──
    ("拉普拉斯锐化",       _sharpen),
    ("强锐化(×2)",         lambda img: _sharpen(_sharpen(img))),
    ("1.5× 超分",          lambda img: _upscale(img, 1.5)),
    ("2× 超分",            lambda img: _upscale(img, 2.0)),

    # ── 倾斜 ──
    ("±3° 校正",           lambda img: _rotate(img, 3)),
    ("−3° 校正",           lambda img: _rotate(img, -3)),
    ("±7° 校正",           lambda img: _rotate(img, 7)),
    ("−7° 校正",           lambda img: _rotate(img, -7)),
    ("±12° 校正",          lambda img: _rotate(img, 12)),
    ("−12° 校正",          lambda img: _rotate(img, -12)),

    # ── 车牌专项 ──
    ("Sobel 边缘增强",     _sobel_edge),
    ("形态学闭运算",       _morph_close),
    ("自适应阈值",         _adaptive_thresh),
    ("蓝牌通道增强",       _blue_enhance),
    ("绿牌通道增强",       _green_enhance),
    ("HSV 色彩增强",       _hsv_enhance),
    ("去噪(双边滤波)",     _denoise),

    # ── 复合 ──
    ("锐化 + CLAHE",       lambda img: _clahe(_sharpen(img))),
    ("边缘 + 蓝牌增强",    lambda img: _blue_enhance(_sobel_edge(img))),
    ("超分 + 锐化",        lambda img: _sharpen(_upscale(img, 2.0))),
    ("超分 + 去噪",        lambda img: _denoise(_upscale(img, 1.5))),
    ("CLAHE + 去噪 + 锐化", lambda img: _sharpen(_denoise(_clahe(img)))),
]


# ═══════════════════════════════════════════════════════
# 核心识别（多参数 + 静默）
# ═══════════════════════════════════════════════════════
def _suppress_stdout(fn, *a, **kw):
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        return fn(*a, **kw)
    finally:
        sys.stdout = old


def _recognize_multi(img):
    """
    用多组参数各识别一次，取所有结果的并集。
    覆盖不同尺寸、不同精细度需求。
    """
    all_results = []
    configs = [
        (15, True),    # 小尺寸 + 字符微调
        (20, True),    # 默认
        (30, True),    # 大尺寸 + 字符微调
        (20, False),   # 关闭字符微调（有时反而更好）
        (40, True),    # 超大尺寸
    ]
    for min_sz, deskew in configs:
        try:
            res = _suppress_stdout(HyperLPR_plate_recognition, img,
                                   minSize=min_sz, charSelectionDeskew=deskew)
            for plate, conf, loc in (res or []):
                all_results.append((plate.strip(), conf, tuple(loc)))
        except Exception:
            pass
    return all_results


# ═══════════════════════════════════════════════════════
# 投票/共识机制
# ═══════════════════════════════════════════════════════
def _vote_and_rank(candidates):
    """
    加权投票排名。
    综合得分 = 最高置信度 × 投票加分 × 位置先验
    位置先验惩罚末尾/稀有字母，使 Q→0 等修正能胜出。
    """
    groups = {}
    for plate, conf in candidates:
        groups.setdefault(plate, []).append(conf)

    ranked = []
    for plate, confs in groups.items():
        best_conf = max(confs)
        freq_bonus = 1.0 + 0.12 * (len(confs) - 1)
        prior     = _position_prior(plate)         # 位置先验
        score     = best_conf * freq_bonus * prior
        ranked.append((plate, min(score, 1.0), len(confs)))

    ranked.sort(key=lambda x: -x[1])
    return ranked


# ═══════════════════════════════════════════════════════
# 单图处理
# ═══════════════════════════════════════════════════════
def process_image(filepath):
    print("\n" + "=" * 56)
    print(f"  {os.path.basename(filepath)}")
    print("=" * 56)

    img = imread_cn(filepath)
    if img is None:
        print("  [跳过] 无法读取")
        return

    raw_candidates = []   # (plate, conf) — 所有候选（含混淆展开）
    seen  = set()
    shown_per_strategy = {}  # strategy_name → best (plate, conf, tag)

    for strategy_name, fn in ENHANCE_STRATEGIES:
        enhanced = fn(img)
        raw_results = _recognize_multi(enhanced)

        if not raw_results:
            continue

        best_for_strategy = None
        for raw_plate, conf, _ in raw_results:
            if not raw_plate:
                continue

            for cand_plate, cand_conf, cand_tag, detail in candidates_from(raw_plate, conf):
                if not cand_plate or cand_plate in seen:
                    continue
                seen.add(cand_plate)

                raw_candidates.append((cand_plate, cand_conf))

                # 记录每个策略的最佳原始检出
                if cand_tag in ("原始", "修正") and (best_for_strategy is None or cand_conf > best_for_strategy[1]):
                    best_for_strategy = (cand_plate, cand_conf, cand_tag, detail, raw_plate)

        if best_for_strategy:
            shown_per_strategy[strategy_name] = best_for_strategy

    # ── 打印：每策略只显示最佳原始检出 ──
    for strategy_name, (plate, conf, tag, detail, raw) in shown_per_strategy.items():
        ok  = "[OK]" if is_chinese_plate(plate) else "[NG]"
        src = f"← {strategy_name}" if strategy_name != "原图" else ""
        if tag == "修正":
            src += " [修正]"
        print(f"  {ok} {plate:12s} {conf:.1%}  {src}")
        if tag == "修正" and detail:
            print(f"       {raw} → {plate}  ({detail})")

    # ── 投票排名 ──
    if raw_candidates:
        ranked = _vote_and_rank(raw_candidates)
        print(f"  {'─'*50}")
        print(f"  投票排名 (含混淆展开 × 位置先验):")
        for i, (plate, score, cnt) in enumerate(ranked[:5], 1):
            marker = "★" if i == 1 else " "
            valid  = "[OK]" if is_chinese_plate(plate) else "[NG]"
            extra  = f"命中 {cnt} 次" if cnt > 1 else ""
            # 标注是否来自混淆展开（不在原始策略检出中）
            from_expansion = plate not in {p for p, *_ in shown_per_strategy.values()}
            note = " [展开]" if from_expansion else ""
            print(f"    {marker} #{i} {plate:12s}  得分:{score:.1%}  {valid}  {extra}{note}")

        best_plate, best_score, best_cnt = ranked[0]
        print()
        if is_chinese_plate(best_plate) and best_cnt >= 1:
            print(f"  >>> 最终结果: {best_plate}  (置信度 {best_score:.1%})")
        elif best_cnt >= 2:
            print(f"  >>> 最终结果: {best_plate}  (置信度 {best_score:.1%})  [格式可疑但高命中]")
        else:
            print(f"  >>> 最终结果: {best_plate}  (置信度 {best_score:.1%})  [低置信度，建议人工复核]")
    else:
        print(f"  [未检测到车牌]")


# ═══════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════
def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"错误: 找不到 {INPUT_DIR}/ 目录")
        return

    images = sorted(
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if os.path.splitext(f)[1].lower() in EXTENSIONS
    )
    if not images:
        print(f"错误: {INPUT_DIR}/ 目录中没有图片文件")
        return

    print(f"  {len(images)} 张图片 | {len(ENHANCE_STRATEGIES)} 种增强 × 5 组参数 = "
          f"{len(ENHANCE_STRATEGIES) * 5} 次识别/图")

    for path in images:
        process_image(path)

    print("\n" + "=" * 56)
    print("  完成")


if __name__ == "__main__":
    main()
