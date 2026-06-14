# 🚗 Chinese License Plate Recognition (HyperLPR)

基于 [HyperLPR](https://github.com/szad670401/HyperLPR) 的中国车牌识别脚本，内置多层容错增强，适用于光照不足、倾斜、模糊、污损等复杂场景。

简体中文 | [English](#english)

---

## ✨ 特性

- **开箱即用** — 图片丢进 `测试照片/` 目录，一键运行
- **28 种图像增强** — CLAHE、Gamma、锐化、超分、旋转校正 …
- **5 组参数扫描** — minSize + deskew 组合覆盖不同尺寸
- **中国车牌校验** — 31 省份 + 格式正则，过滤非中国牌照
- **易混淆字符修正** — `I→1` `O→0` `Q→0` `D→0` 等 OCR 纠正
- **位置先验** — 末尾稀有字母自动惩罚，打破平票
- **加权投票** — 多策略共识命中加分
- **中文路径兼容** — Windows GBK 路径无报错

## 📦 安装

```bash
# Python 3.7+
pip install hyperlpr opencv-python numpy

# ⚠️ 兼容性修复（hyperlpr 0.0.2 依赖已废弃 API）
# 修复脚本自动处理，详见下方说明
```

> **兼容性说明**: hyperlpr 0.0.2 使用 `np.int` (NumPy ≥1.24 已移除) 和 `cv2.estimateRigidTransform` (OpenCV ≥4.5 已移除)。本项目运行时会自动检测并指导修复。

## 🚀 快速开始

```bash
# 1. 将要识别的图片放入 测试照片/ 目录
#    支持 .jpg .jpeg .png .bmp

# 2. 运行
python plate_recognition.py
```

**输出示例：**

```
  6 张图片 | 28 种增强 × 5 组参数 = 140 次识别/图

========================================================
  1.jpg
========================================================
  [OK] 粤BA20950     79.1%
  [OK] 粤BA2D95D     78.6%  ← CLAHE 弱光增强
  [OK] 粤B120950     78.5%  ← Gamma 提亮
  [OK] 粤BA2095Q     82.6%  ← Sobel 边缘增强
  ──────────────────────────────────────────────────
  投票排名 (含混淆展开 × 位置先验):
    ★ #1 粤BA20950      得分:79.1%  [OK]
      #2 粤B120950      得分:78.5%  [OK]
      #3 鲁B12D950      得分:76.7%  [OK]
      #4 鲁BD20950      得分:76.5%  [OK]
      #5 皖BB1D5D2      得分:76.1%  [OK]

  >>> 最终结果: 粤BA20950  (置信度 79.1%)
```

## 🛡️ 容错机制

### 图像增强管线（28 种策略）

| 场景 | 策略 |
|---|---|
| ☀️ 光线不足 | CLAHE 弱光增强、Gamma 提亮、自动亮度 |
| 🌫️ 图像模糊 | 拉普拉斯锐化、强锐化 ×2 |
| 📐 倾斜拍摄 | ±3° / ±7° / ±12° 旋转校正 |
| 🔍 低分辨率 | 1.5× / 2× 超分放大 |
| 🟦 蓝牌褪色 | 蓝牌通道增强 |
| 🟩 新能源绿牌 | 绿牌通道增强 |
| 🏚️ 老旧褪色 | HSV 色彩增强 |
| 🌑 光照不均 | 自适应阈值 |
| 🩹 污损字符 | 形态学闭运算、双边滤波去噪 |
| 🧩 复合劣化 | 锐化+CLAHE、边缘+蓝牌、超分+去噪+锐化 |

### 易混淆字符修正

```
原始检出      修正后        原因
──────────────────────────────────
京AI2345  →  京A12345     I→1 (车牌不用I)
沪0A2345  →  沪DA2345     城市码0→D (必须字母)
粤BA2095Q →  粤BA20950    Q→0 (末尾稀有字母)
```

**三层修正漏斗：**

| 层级 | 机制 | 触发条件 |
|---|---|---|
| ① 规则修正 | 非法字符强制替换（I/O 不在车牌字符集） | 始终 |
| ② 混淆展开 | 仅对易混字符生成替代候选 | Q/X/Y/Z/0/1/5/6/8/B/G |
| ③ 位置先验 | 末尾稀有字母扣分 + 投票共识 | 多候选平票时 |

### 投票排名

综合得分 = 最高置信度 × 命中次数加分 × 位置先验因子

---

## 📁 项目结构

```
车牌识别/
├── plate_recognition.py   # 主脚本
├── README.md              # 本文件
└── 测试照片/              # 图片放入此处
    ├── 1.jpg
    └── 2.jpg
```

## ⚙️ 配置

修改 `plate_recognition.py` 顶部的常量：

```python
INPUT_DIR  = "测试照片"     # 图片目录
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}  # 支持格式
```

调整体量（在 `_recognize_multi` 中）：

```python
configs = [
    (15, True),   # minSize, charSelectionDeskew
    (20, True),
    ...
]
```

## 📄 License

MIT

---

## English

A Chinese license plate recognition script based on HyperLPR with multi-layer fault tolerance.

### Key Features
- 28 image enhancement strategies covering low light, blur, tilt, noise, low resolution
- Chinese plate format validation (31 provinces, 7/8 char format)
- OCR confusion correction (`I→1`, `O→0`, `Q→0`, etc.)
- Weighted voting across enhancement strategies
- Windows Chinese-path compatible

### Usage
1. Place images in `测试照片/`
2. Run `python plate_recognition.py`
3. Results printed to terminal with confidence scores and correction details
