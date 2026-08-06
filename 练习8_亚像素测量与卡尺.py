# 第8课 练习8：亚像素测量 + 卡尺工具
# 目标：把圆直径测量从"1像素精度"提升到"0.1像素精度"
# 原理：卡尺扫描（沿法线找灰度突变）+ 抛物线插值（亚像素定位）+ 最小二乘圆拟合
import cv2
import numpy as np
import os

def imwrite_unicode(path, img):
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(path)
    return ok

def gray_at(img, x, y):
    """双线性插值：在任意小数坐标处取灰度值（亚像素采样的基础）"""
    h, w = img.shape[:2]
    x = max(0.0, min(w - 1.0, x))
    y = max(0.0, min(h - 1.0, y))
    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
            + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)

def caliper_edges(gray, cx, cy, r0, r1, n=36):
    """卡尺工具核心：沿圆周 n 个方向扫描灰度梯度，抛物线插值得到亚像素边缘点
    cx,cy: 粗略圆心  r0,r1: 扫描半径范围  n: 卡尺数量"""
    pts = []
    for k in range(n):
        th = 2 * np.pi * k / n
        dx, dy = np.cos(th), np.sin(th)
        ts = np.arange(r0, r1, 0.5)                       # 沿法线方向取点
        vals = np.array([gray_at(gray, cx + dx * t, cy + dy * t) for t in ts])
        g = np.abs(np.diff(vals.astype(np.float32)))      # 灰度梯度
        i = int(np.argmax(g))
        if i <= 0 or i >= len(g) - 1:
            continue
        g0, g1, g2 = g[i - 1], g[i], g[i + 1]             # 抛物线插值亚像素位置
        den = g0 - 2 * g1 + g2
        delta = 0.5 * (g0 - g2) / den if abs(den) > 1e-9 else 0.0
        t = (ts[i] + ts[i + 1]) * 0.5 + delta   # 梯度属于区间中点，再加抛物线亚像素偏移
        pts.append((cx + dx * t, cy + dy * t))
    return np.array(pts)

def fit_circle(pts):
    """最小二乘圆拟合（Kasa代数法）：对所有边缘点整体拟合一个圆，抗单点噪声"""
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x * x + y * y
    a, c, d = np.linalg.lstsq(A, b, rcond=None)[0]
    r = np.sqrt(d + a * a + c * c)
    return a, c, r

def measure_pixel(gray):
    """整像素方法（旧方法）：阈值 + 取反 + 轮廓 + 最小外接圆"""
    _, bin_ = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    part = cv2.bitwise_not(bin_)          # 背光：工件(黑)取反变白，便于找轮廓
    contours, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = max(contours, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(big)
    return x, y, r

def measure_subpixel(gray):
    """亚像素方法（新方法）：粗定位 + 卡尺 + 最小二乘圆"""
    cx, cy, r = measure_pixel(gray)                       # 第一步：整像素粗定位
    pts = caliper_edges(gray, cx, cy, r - 20, r + 20)     # 第二步：卡尺精测量
    return fit_circle(pts), pts

# ---------- 实验1：真实垫片图 ----------
path = "练习6_标定垫片.png"
if os.path.exists(path):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        gray = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    print(f"实验1 素材：{path} 尺寸 {gray.shape[1]}x{gray.shape[0]}")
else:
    gray = np.full((300, 400), 245, dtype=np.uint8)       # 没有素材就画一个
    cv2.circle(gray, (200, 150), 90, 0, -1)
    cv2.circle(gray, (200, 150), 40, 245, -1)
    imwrite_unicode(path, gray)
    print("实验1 素材：程序生成的垫片图（200,150 半径90）")

pcx, pcy, pr = measure_pixel(gray)
(scx, scy, sr), sedges = measure_subpixel(gray)
print(f"\n整像素: 中心({pcx:.1f},{pcy:.1f}) 直径 {2*pr:.1f} px")
print(f"亚像素: 中心({scx:.2f},{scy:.2f}) 直径 {2*sr:.2f} px   (卡尺点数 {len(sedges)})")

# ---------- 可视化 ----------
vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
for (x, y) in sedges:                                     # 卡尺边缘点（红）
    cv2.circle(vis, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)
cv2.circle(vis, (int(round(scx)), int(round(scy))), int(round(sr)), (0, 255, 0), 2)   # 亚像素圆（绿）
cv2.circle(vis, (int(round(pcx)), int(round(pcy))), int(round(pr)), (255, 0, 0), 1)   # 整像素圆（蓝）
cv2.putText(vis, f"pixel: {2*pr:.1f}px", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
cv2.putText(vis, f"subpixel: {2*sr:.2f}px", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
imwrite_unicode("演示8_亚像素测量.png", vis)
print("可视化已保存：演示8_亚像素测量.png")

# ---------- 实验2：光源波动实验（工业上最常遇到的干扰） ----------
def make_aa_washer(r_true, center=(200.0, 150.0)):
    """用浮点距离生成带平滑边缘的圆，真值半径精确已知 r_true（模拟镜头弥散）"""
    yy, xx = np.mgrid[0:300, 0:400].astype(np.float32)
    d = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    w = 2.0                                   # 边缘过渡宽度(px)，模拟真实镜头弥散
    g = 245.0 * 0.5 * (1.0 + np.tanh((d - r_true) / (w / 2.0)))
    return np.clip(g, 0, 255).astype(np.uint8)

rng = np.random.default_rng(7)
print("\n实验2：光源波动实验——20次亮度±20漂移 + 噪声σ=6，真值半径88.5~91.5px：")
d_px, d_sp = [], []
for _ in range(20):
    r_true = rng.uniform(88.5, 91.5)
    img = make_aa_washer(r_true)
    bright = rng.uniform(-20, 20)             # 光源波动：整体亮度漂移
    img = np.clip(img.astype(np.int16) + bright, 0, 255).astype(np.uint8)
    noise = np.clip(rng.normal(0, 6, img.shape), -18, 18).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 1.5)  # 先降噪再卡尺（真实工业流程）
    _, _, rp = measure_pixel(img)
    (_, _, rs), _ = measure_subpixel(img)
    d_px.append(abs(2 * rp - 2 * r_true))
    d_sp.append(abs(2 * rs - 2 * r_true))
d_px, d_sp = np.array(d_px), np.array(d_sp)
print(f"  整像素(固定阈值): 平均误差 {d_px.mean():.3f}px  最大 {d_px.max():.3f}px")
print(f"  亚像素(梯度峰值): 平均误差 {d_sp.mean():.3f}px  最大 {d_sp.max():.3f}px")
print(f"  结论: 亚像素不受亮度漂移影响，误差缩小 {d_px.mean()/d_sp.mean():.0f} 倍")
print("  (换算到 0.05mm/px: 整像素≈{:.4f}mm, 亚像素≈{:.4f}mm)".format(d_px.mean()*0.05, d_sp.mean()*0.05))
