# 第6课 视觉端：拍照 -> 检测 + 测量 -> 把"判断 + 结果"发给PLC
# 运行前提：先启动 练习6_模拟PLC端.py
import socket
import time
import cv2
import numpy as np

rng = np.random.default_rng(7)

def imwrite_unicode(path, img):
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(path)
    return ok

# ---------- 标准垫片（背光）用于标定 ----------
def make_washer():
    w = np.full((300, 400), 245, dtype=np.uint8)
    cv2.circle(w, (200, 150), 90, 0, -1)
    cv2.circle(w, (200, 150), 40, 245, -1)
    return w

# ---------- 缺陷工件（前光，表面带纹理） ----------
def make_part(defect=False):
    p = np.full((300, 300), 60, dtype=np.uint8)
    cv2.circle(p, (150, 150), 120, 130, -1)
    t = np.clip(rng.normal(0, 10, p.shape), -25, 25).astype(np.int16)
    p = np.clip(p.astype(np.int16) + t, 0, 255).astype(np.uint8)
    if defect:
        cv2.circle(p, (95, 110), 10, 230, -1)
        cv2.circle(p, (200, 190), 6, 210, -1)
    return p

# ---------- 检测缺陷：高阈值(190)找异常亮点 ----------
def check_defect(p, th=190):
    _, bin_p = cv2.threshold(p, th, 255, cv2.THRESH_BINARY)
    area = np.count_nonzero(bin_p == 255)
    return area, area > 80

# ---------- 分割测量：低阈值(100)把整个工件切出来，测直径 ----------
def measure_part(p, precision):
    _, b = cv2.threshold(p, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = max(contours, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(big)
    d_px = 2 * r
    return d_px, d_px * precision

# ---------- 第1步：标定（像素 -> 毫米） ----------
print("========== 第1步：标定（像素 -> 毫米） ==========")
washer = make_washer()
_, b = cv2.threshold(washer, 127, 255, cv2.THRESH_BINARY)
mask = cv2.bitwise_not(b)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
big = max(contours, key=cv2.contourArea)
(x, y), r = cv2.minEnclosingCircle(big)
washer_px = 2 * r
real_mm = 9.0                     # 标准件真实直径（已知）
precision = real_mm / washer_px   # 每个像素 = 多少毫米
print(f"标准垫片：图像直径 {washer_px:.1f} 像素，真实直径 {real_mm} mm")
print(f"标定精度：{precision:.4f} mm/像素  （以后：像素数 x 精度 = 毫米）")
imwrite_unicode("练习6_标定垫片.png", washer)

# ---------- 第2步：检测 5 个工件，把"判断+结果"发给 PLC ----------
print("========== 第2步：检测 5 个工件 ==========")
parts = [make_part(False), make_part(True), make_part(False), make_part(False), make_part(True)]

PLC_ADDR = ("127.0.0.1", 5001)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(PLC_ADDR)
print(f"已连接到模拟PLC {PLC_ADDR}")

for i, p in enumerate(parts, 1):
    time.sleep(0.5)               # 模拟拍照节拍
    area, ng = check_defect(p)
    d_px, d_mm = measure_part(p, precision)
    result = "NG" if ng else "OK"
    msg = f"{result},{d_mm:.2f}"
    sock.sendall((msg + "\n").encode("utf-8"))
    print(f"工件{i}: 缺陷面积={area} -> 判断={result}，直径={d_mm:.2f}mm（{d_px:.0f}px）-> 已发送")
    cv2.imshow(f"正在检测 工件{i}（{result}）", p)
    cv2.waitKey(400)              # 每个工件显示0.4秒

cv2.destroyAllWindows()
sock.close()
print("5个工件检测完毕，连接已关闭")