# 第4课 练习4：图像处理基础全流程演示（窗口10秒后自动关闭）
import cv2
import numpy as np

def imwrite_unicode(path, img):
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(path)
    return ok

rng = np.random.default_rng(42)

# ---------- 素材1：背光垫片（第2课的老朋友） ----------
back = np.full((300, 400), 245, dtype=np.uint8)
cv2.circle(back, (200, 150), 90, 0, -1)
cv2.circle(back, (200, 150), 40, 245, -1)

# ---------- 素材2：带椒盐噪声的前光图 ----------
front = np.full((300, 400), 60, dtype=np.uint8)
noise = np.clip(rng.normal(0, 14, front.shape), -30, 30).astype(np.int16)
front = np.clip(front.astype(np.int16) + noise, 0, 255).astype(np.uint8)
cv2.circle(front, (200, 150), 90, 140, -1)
cv2.circle(front, (200, 150), 40, 40, -1)
cv2.line(front, (140, 100), (240, 210), 200, 3)
front[rng.random(front.shape) < 0.005] = 255
front[rng.random(front.shape) < 0.005] = 0

# ---------- 1. 直方图 ----------
hist = cv2.calcHist([back], [0], None, [256], [0, 256])
hist_img = np.zeros((300, 400), dtype=np.uint8)
hist_norm = cv2.normalize(hist, None, 0, 280, cv2.NORM_MINMAX).flatten()
for i in range(256):
    x = int(i * 400 / 256)
    cv2.line(hist_img, (x, 300), (x, 300 - int(hist_norm[i])), 255, 1)

# ---------- 2. 二值化 ----------
_, bin_img = cv2.threshold(back, 127, 255, cv2.THRESH_BINARY)

# ---------- 3. 中值滤波对比 ----------
_, raw_bin = cv2.threshold(front, 100, 255, cv2.THRESH_BINARY)
median = cv2.medianBlur(front, 5)
_, med_bin = cv2.threshold(median, 100, 255, cv2.THRESH_BINARY)

# ---------- 4. 形态学开运算 ----------
kernel = np.ones((5, 5), np.uint8)
opened = cv2.morphologyEx(raw_bin, cv2.MORPH_OPEN, kernel)

# ---------- 5. Canny 边缘 ----------
edges = cv2.Canny(back, 50, 150)

# ---------- 6. 轮廓 + 特征（测量） ----------
part_mask = cv2.bitwise_not(bin_img)
contours, _ = cv2.findContours(part_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
big = max(contours, key=cv2.contourArea)
area = cv2.contourArea(big)
peri = cv2.arcLength(big, True)
M = cv2.moments(big)
cx = M["m10"] / M["m00"]
cy = M["m01"] / M["m00"]
(x, y), r = cv2.minEnclosingCircle(big)

# 把结果画在垫片图上：红=轮廓，绿=中心，蓝=拟合圆
result = cv2.cvtColor(back, cv2.COLOR_GRAY2BGR)
cv2.drawContours(result, [big], -1, (0, 0, 255), 3)              # 红色：真实边缘
cv2.circle(result, (int(cx), int(cy)), 6, (0, 255, 0), -1)       # 绿色：圆心
cv2.circle(result, (int(x), int(y)), int(r), (255, 0, 0), 2)     # 蓝色：拟合圆
cv2.putText(result, f"D={2*r:.1f}px", (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.putText(result, f"Center=({cx:.1f},{cy:.1f})", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
imwrite_unicode("演示_测量结果.png", result)

print("========== 图像处理流程结果 ==========")
print(f"面积: {area:.0f} 像素^2（理论外圆 pi*90^2 ≈ 25447，内孔是内部轮廓稍后学）")
print(f"周长: {peri:.1f} 像素（理论 2*pi*90 ≈ 565.5）")
print(f"中心: ({cx:.1f}, {cy:.1f})（理论 (200, 150)）")
print(f"外圆直径: {2*r:.1f} 像素（理论 180）")
print("已保存：演示_测量结果.png（红=轮廓 绿=中心 蓝=拟合圆）")

# ---------- 图例说明面板 ----------
legend = np.zeros((300, 400, 3), dtype=np.uint8)
cv2.putText(legend, "COLOR LEGEND", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
cv2.line(legend, (25, 90), (100, 90), (0, 0, 255), 4)
cv2.putText(legend, "red  = real edge", (115, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
cv2.circle(legend, (62, 150), 7, (0, 255, 0), -1)
cv2.putText(legend, "green = center", (115, 156), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
cv2.circle(legend, (62, 210), 35, (255, 0, 0), 2)
cv2.putText(legend, "blue = fitted circle", (115, 216), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
cv2.putText(legend, "D=180.0px", (10, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# ---------- 拼图显示 ----------

def show_fit(title, img, max_h=700, max_w=1200):
    """自动缩放窗口，避免图像比屏幕高导致底部看不到（保存的图仍是原图）"""
    h, w = img.shape[:2]
    scale = min(1.0, max_h / h, max_w / w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, img.shape[1], img.shape[0])
    cv2.imshow(title, img)

def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img

def to_bgr(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

grid = np.vstack([
    np.hstack([label(to_bgr(back), "1 ORIGINAL"), label(to_bgr(hist_img), "2 HISTOGRAM")]),
    np.hstack([label(to_bgr(bin_img), "3 BINARY"), label(to_bgr(median), "4 MEDIAN FILTER")]),
    np.hstack([label(to_bgr(opened), "5 OPEN (CLEAN)"), label(to_bgr(edges), "6 CANNY EDGE")]),
    np.hstack([label(result, "7 MEASURE RESULT"), label(legend, "8 COLOR LEGEND")]),
])
imwrite_unicode("演示_图像处理流程.png", grid)
show_fit("图像处理流程（7=测量结果 8=颜色图例）", grid)
cv2.waitKey(10000)
cv2.destroyAllWindows()
print("练习4演示结束")