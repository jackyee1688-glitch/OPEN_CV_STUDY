# 第5课 练习5：五大经典视觉应用演示（窗口12秒后自动关闭）
import cv2
import numpy as np
import qrcode

def imwrite_unicode(path, img):
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(path)
    return ok

def to_bgr(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

rng = np.random.default_rng(42)

print("========== 应用1：模板匹配定位（找东西在哪） ==========")
# 传送带场景：3个元件，只有中间那个带小孔（独有特征）
scene = np.full((300, 400), 245, dtype=np.uint8)
cv2.circle(scene, (80, 80), 30, 0, -1)
cv2.circle(scene, (200, 160), 30, 0, -1)
cv2.circle(scene, (320, 80), 30, 0, -1)
cv2.circle(scene, (200, 160), 12, 245, -1)          # 元件2的小孔（特征）
template = scene[130:190, 170:230]                  # 以元件2为中心裁60x60当模板
res = cv2.matchTemplate(scene, template, cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(res)
tx, ty = max_loc
cx, cy = tx + 30, ty + 30
print(f"最佳相似度: {max_val:.3f}  定位中心: ({cx}, {cy})  (理论 200,160)")
match_img = to_bgr(scene)
cv2.rectangle(match_img, (tx, ty), (tx + 60, ty + 60), (0, 0, 255), 2)
cv2.circle(match_img, (cx, cy), 5, (0, 255, 0), -1)
imwrite_unicode("演示5_模板匹配.png", match_img)

print("========== 应用2：几何测量（量多大） ==========")
meas = np.full((300, 400), 245, dtype=np.uint8)
cv2.circle(meas, (120, 150), 50, 0, -1); cv2.circle(meas, (120, 150), 20, 245, -1)
cv2.circle(meas, (280, 150), 30, 0, -1); cv2.circle(meas, (280, 150), 12, 245, -1)
_, m_bin = cv2.threshold(meas, 127, 255, cv2.THRESH_BINARY)
m_mask = cv2.bitwise_not(m_bin)
m_contours, _ = cv2.findContours(m_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
m_contours = sorted(m_contours, key=cv2.contourArea, reverse=True)

def circle_info(c):
    M = cv2.moments(c)
    cxx = M["m10"] / M["m00"]; cyy = M["m01"] / M["m00"]
    (x, y), r = cv2.minEnclosingCircle(c)
    return cxx, cyy, r

cx1, cy1, r1 = circle_info(m_contours[0])
cx2, cy2, r2 = circle_info(m_contours[1])
dist = np.hypot(cx1 - cx2, cy1 - cy2)
print(f"垫片1直径: {2*r1:.1f}px (理论100)  垫片2直径: {2*r2:.1f}px (理论60)")
print(f"中心距: {dist:.1f}px (理论160)")
meas_img = to_bgr(meas)
cv2.drawContours(meas_img, m_contours[:2], -1, (0, 0, 255), 2)
cv2.line(meas_img, (int(cx1), int(cy1)), (int(cx2), int(cy2)), (0, 255, 0), 2)
cv2.putText(meas_img, f"dist={dist:.1f}px", (150, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
imwrite_unicode("演示5_几何测量.png", meas_img)

print("========== 应用3：计数（数几个） ==========")
cnt_img = np.full((300, 400), 245, dtype=np.uint8)
positions = [(40, 40), (110, 40), (180, 40), (210, 110), (45, 180), (115, 180), (185, 180)]
for x, y in positions:
    cv2.circle(cnt_img, (x, y), 22, 0, -1)
_, c_bin = cv2.threshold(cnt_img, 127, 255, cv2.THRESH_BINARY)
c_mask = cv2.bitwise_not(c_bin)
c_contours, _ = cv2.findContours(c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
areas = [cv2.contourArea(c) for c in c_contours]
print(f"识别数量: {len(c_contours)} (理论 {len(positions)})")
print(f"单粒平均面积: {np.mean(areas):.0f} 像素^2 (理论 pi*22^2 ≈ 1521)")
cnt_vis = to_bgr(cnt_img)
for c in c_contours:
    M = cv2.moments(c)
    cv2.drawContours(cnt_vis, [c], -1, (0, 0, 255), 2)
    cv2.circle(cnt_vis, (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])), 3, (0, 255, 0), -1)
cv2.putText(cnt_vis, f"count={len(c_contours)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
imwrite_unicode("演示5_计数.png", cnt_vis)

print("========== 应用4：缺陷检测（好不好） ==========")
def make_part(defect=False):
    p = np.full((300, 300), 60, dtype=np.uint8)
    cv2.circle(p, (150, 150), 120, 130, -1)                 # 先画金属表面
    t = np.clip(rng.normal(0, 10, p.shape), -25, 25).astype(np.int16)
    p = np.clip(p.astype(np.int16) + t, 0, 255).astype(np.uint8)  # 再加表面纹理波动（更真实）
    if defect:
        cv2.circle(p, (95, 110), 10, 230, -1)    # 反光亮点 = 划痕/脏污
        cv2.circle(p, (200, 190), 6, 210, -1)    # 第二个小缺陷
    return p

def check_defect(p):
    _, bin_p = cv2.threshold(p, 190, 255, cv2.THRESH_BINARY)
    area = np.count_nonzero(bin_p == 255)
    return area, area > 80    # 异常亮斑面积超80像素 → NG

ok_part = make_part(False)
ng_part = make_part(True)
ok_area, ok_ng = check_defect(ok_part)
ng_area, ng_ng = check_defect(ng_part)
print(f"好品: 异常亮斑面积={ok_area} -> {'NG' if ok_ng else 'OK'}")
print(f"坏品: 异常亮斑面积={ng_area} -> {'NG' if ng_ng else 'OK'}")
defect_vis = np.hstack([to_bgr(ok_part), to_bgr(ng_part)])
cv2.putText(defect_vis, f"GOOD -> {"NG" if ok_ng else "OK"}  area={ok_area}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if ok_ng else (0, 255, 0), 2)
cv2.putText(defect_vis, f"DEFECT -> {"NG" if ng_ng else "OK"}  area={ng_area}", (310, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if ng_ng else (0, 255, 0), 2)
imwrite_unicode("演示5_缺陷检测.png", defect_vis)

print("========== 应用5：二维码识别（是什么） ==========")
qr = qrcode.QRCode(border=4)
qr.add_data("P/N:2026-0803 BATCH:42")
qr.make(fit=True)
qr_pil = qr.make_image(fill_color="black", back_color="white").convert("L")
qr_np = np.array(qr_pil)
qr_big = cv2.resize(qr_np, (150, 150), interpolation=cv2.INTER_NEAREST)
label_img = np.full((300, 400), 245, dtype=np.uint8)
label_img[75:225, 125:275] = qr_big
det = cv2.QRCodeDetector()
data, pts, _ = det.detectAndDecode(label_img)
print(f"识别到的二维码内容: {data}")
qr_vis = to_bgr(label_img)
if pts is not None:
    pts = pts.reshape(-1, 2).astype(int)
    cv2.polylines(qr_vis, [pts], True, (0, 255, 0), 3)
imwrite_unicode("演示5_二维码.png", qr_vis)

print("========== 汇总 ==========")
print("定位=模板匹配找坐标 | 测量=轮廓+拟合算尺寸 | 计数=数轮廓")
print("缺陷=阈值找异常区域 | 读码=直接解码")

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

grid = np.vstack([
    np.hstack([label(match_img, "1 TEMPLATE MATCH"), label(meas_img, "2 MEASURE")]),
    np.hstack([label(cnt_vis, "3 COUNT"), label(qr_vis, "5 QR CODE")]),
    np.hstack([label(defect_vis, "4 DEFECT OK/NG"), np.zeros((300, 200, 3), np.uint8)]),
])
imwrite_unicode("演示5_汇总.png", grid)
show_fit("五大应用（1定位 2测量 3计数 4缺陷 5读码）", grid)
cv2.waitKey(22000)
cv2.destroyAllWindows()
print("练习5演示结束")