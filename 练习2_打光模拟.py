# 第2课 练习2：打光效果模拟（窗口8秒后自动关闭）
# 同一个垫片工件：前光 vs 背光，比较二值化效果
import cv2
import numpy as np

def imwrite_unicode(path, img):
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(path)
    return ok

rng = np.random.default_rng(42)

# ---------- 场景1：前光（明场）----------
# 光源和相机在同一侧，看到的是表面反射：金属亮、内孔暗、还有划痕
front = np.full((300, 400), 60, dtype=np.uint8)                    # 暗背景
noise = np.clip(rng.normal(0, 14, front.shape), -30, 30).astype(np.int16)
front = np.clip(front.astype(np.int16) + noise, 0, 255).astype(np.uint8)
cv2.circle(front, (200, 150), 90, 140, -1)    # 金属垫片表面（反光→偏亮）
cv2.circle(front, (200, 150), 40, 40, -1)     # 内孔（暗）
cv2.line(front, (140, 100), (240, 210), 200, 3)   # 划痕1（反光→高亮）
cv2.line(front, (250, 90), (265, 200), 175, 2)    # 划痕2

# ---------- 场景2：背光（剪影）----------
# 光源在工件背后：工件挡住光→纯黑剪影，内孔透光→亮，边缘锐利
back = np.full((300, 400), 245, dtype=np.uint8)
cv2.circle(back, (200, 150), 90, 0, -1)       # 工件剪影
cv2.circle(back, (200, 150), 40, 245, -1)     # 内孔透光

# ---------- 同一套阈值分割算法，两个场景各跑一遍 ----------
_, front_bin = cv2.threshold(front, 100, 255, cv2.THRESH_BINARY)
_, back_bin = cv2.threshold(back, 127, 255, cv2.THRESH_BINARY)

# ---------- 量化"分割干净度" ----------
yy, xx = np.mgrid[0:300, 0:400]
d = np.sqrt((xx - 200) ** 2 + (yy - 150) ** 2)
part_area = (d <= 90) & (d > 40)     # 垫片实体（环形区域）
hole_area = d <= 40                  # 内孔
bg_area = d > 90                     # 背景

# 前光：实体应白、背景和孔应黑，数一数判错的像素
err_front = np.count_nonzero(front_bin[bg_area] == 255) + np.count_nonzero(front_bin[hole_area] == 255)
# 背光：实体应黑，数一数判错的像素
err_back = np.count_nonzero(back_bin[part_area] == 255)

print("========== 打光模拟结果 ==========")
print(f"前光：背景和内孔被误判为白色的像素 = {err_front}")
print(f"背光：工件实体被误判为白色的像素   = {err_back}")
print(f"结论：背光分割干净（误判 0），前光被表面纹理干扰（误判 {err_front}）")
print("        所以尺寸测量首选背光，缺陷检测才用前光/暗场。")

# ---------- 生成对比图：上排前光，下排背光 ----------

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
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return img

front_c = cv2.cvtColor(front, cv2.COLOR_GRAY2BGR)
front_bin_c = cv2.cvtColor(front_bin, cv2.COLOR_GRAY2BGR)
back_c = cv2.cvtColor(back, cv2.COLOR_GRAY2BGR)
back_bin_c = cv2.cvtColor(back_bin, cv2.COLOR_GRAY2BGR)

row1 = np.hstack([label(front_c, "FRONT LIGHT"), label(front_bin_c, "FRONT BINARY")])
row2 = np.hstack([label(back_c, "BACK LIGHT"), label(back_bin_c, "BACK BINARY")])
grid = np.vstack([row1, row2])
imwrite_unicode("演示_打光对比.png", grid)

show_fit("打光对比：上=前光 下=背光", grid)
print("窗口显示中，8秒后自动关闭……")
cv2.waitKey(8000)
cv2.destroyAllWindows()
print("练习2演示结束")