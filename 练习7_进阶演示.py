# 第7课 练习7：进阶方向体验 —— 生成数据集 + 3D深度图（窗口10秒后自动关闭）
import cv2
import numpy as np
import os

rng = np.random.default_rng(2026)

def imwrite_unicode(path, img):
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(path)
    return ok

def show_fit(title, img, max_h=700, max_w=1200):
    h, w = img.shape[:2]
    scale = min(1.0, max_h / h, max_w / w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, img.shape[1], img.shape[0])
    cv2.imshow(title, img)

# ========== 第一部分：生成深度学习训练数据集 ==========
print("========== 第一部分：生成训练数据集 ==========")
print("深度学习要'喂'数据：好品和坏品的图片越多越多样，模型越准")

def make_part(defect=False):
    p = np.full((300, 300), 60, dtype=np.uint8)
    cv2.circle(p, (150, 150), 120, 130, -1)
    t = np.clip(rng.normal(0, 10, p.shape), -25, 25).astype(np.int16)
    p = np.clip(p.astype(np.int16) + t, 0, 255).astype(np.uint8)
    if defect:
        n = rng.integers(1, 4)                    # 1~3 个缺陷，位置大小随机
        for _ in range(n):
            ang = rng.uniform(0, 2 * np.pi)
            rad = rng.integers(0, 80)
            cx = int(150 + rad * np.cos(ang))
            cy = int(150 + rad * np.sin(ang))
            rr = int(rng.integers(6, 15))
            val = int(rng.integers(200, 240))
            cv2.circle(p, (cx, cy), rr, val, -1)
    return p

os.makedirs("数据集/OK", exist_ok=True)
os.makedirs("数据集/NG", exist_ok=True)
for i in range(50):
    imwrite_unicode(f"数据集/OK/ok_{i:03d}.png", make_part(False))
for i in range(50):
    imwrite_unicode(f"数据集/NG/ng_{i:03d}.png", make_part(True))
print("已生成：好品 50 张（数据集/OK）+ 坏品 50 张（数据集/NG）")
print("真实项目里，这一步是用相机拍真工件，然后人工分好类")

# 拼一张样品预览：2 好品 + 2 坏品
s = [make_part(False), make_part(False), make_part(True), make_part(True)]
samp = np.vstack([np.hstack([s[0], s[1]]), np.hstack([s[2], s[3]])])
imwrite_unicode("演示7_数据集样本.png", samp)

# ========== 第二部分：3D 深度图模拟 ==========
print("========== 第二部分：3D深度图模拟 ==========")
depth = np.full((300, 400), 100, dtype=np.int16)
yy, xx = np.mgrid[0:300, 0:400]
d = np.sqrt((xx - 200) ** 2 + (yy - 150) ** 2)
dome = np.clip(60 - 0.6 * d, 0, 60)               # 中间凸起的工件表面
depth = depth + dome
dd = np.sqrt((xx - 280) ** 2 + (yy - 90) ** 2)    # 一个凹陷缺陷（坑）
depth = np.where(dd < 18, depth - 40, depth)
depth = np.clip(depth, 0, 255).astype(np.uint8)

# 对比：普通 2D 灰度图里这个坑几乎看不见
flat = np.full((300, 400), 130, dtype=np.uint8)
t = np.clip(rng.normal(0, 8, flat.shape), -20, 20).astype(np.int16)
flat = np.clip(flat.astype(np.int16) + t, 0, 255).astype(np.uint8)

depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
depth_gray = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
flat_bgr = cv2.cvtColor(flat, cv2.COLOR_GRAY2BGR)
print("深度图：灰度值=高度（亮=高 暗=低），凹陷缺陷一眼可见")
print("2D 灰度图：同样的表面，坑几乎看不见 —— 这就是3D视觉的价值")

def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img

depth_show = np.hstack([label(depth_gray, "3D DEPTH"), label(depth_color, "3D COLOR"), label(flat_bgr, "2D INTENSITY")])
imwrite_unicode("演示7_深度图对比.png", depth_show)
samp_bgr = cv2.cvtColor(samp, cv2.COLOR_GRAY2BGR)
samp_bgr = cv2.resize(samp_bgr, (1200, 600), interpolation=cv2.INTER_AREA)
show_fit("数据集样本 + 3D深度图对比", np.vstack([samp_bgr, depth_show]))
cv2.waitKey(10000)
cv2.destroyAllWindows()
print("练习7演示结束")