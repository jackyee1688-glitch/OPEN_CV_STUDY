# 毕业项目：垫片尺寸测量 + 缺陷检测 + PLC 分拣
# 运行前提：先启动 练习6_模拟PLC端.py（模拟PLC监听 5001 端口）
import socket
import time
import os
import cv2
import numpy as np

rng = np.random.default_rng(2026)

MM_PER_PX = 0.05        # 标定：180px = 9.00mm
REF_DIAM_MM = 9.00      # 标准直径
TOL_MM = 0.10           # 直径公差
DEFECT_LIMIT = 80       # 缺陷面积上限
N_SAMPLES = 100
PLC_ADDR = ("127.0.0.1", 5001)

def imwrite_unicode(path, img):
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(path)
    return ok

# ---------- 1. 参考垫片与模板（相当于开机标定） ----------
def make_reference():
    w = np.full((300, 400), 245, dtype=np.uint8)
    cv2.circle(w, (200, 150), 90, 0, -1)
    cv2.circle(w, (200, 150), 40, 245, -1)
    cv2.rectangle(w, (282, 142), (298, 158), 245, -1)   # 右侧定位缺口
    return w

ref = make_reference()
template = ref[70:230, 120:280]      # 160x160 模板
imwrite_unicode("项目_参考垫片.png", ref)

# ---------- 2. 生成 100 个样品（含真值，模拟产线） ----------
def make_sample():
    cx = int(rng.uniform(170, 230))          # 位置随机
    cy = int(rng.uniform(120, 180))
    r_out = rng.uniform(88, 92)              # 直径 8.8~9.2mm 随机
    has_defect = rng.random() < 0.25
    a = rng.uniform(0, 2 * np.pi)            # 缺口角度随机
    nx = int(cx + np.cos(a) * r_out)
    ny = int(cy + np.sin(a) * r_out)
    # 背光视图（测量用）
    back = np.full((300, 400), 245, dtype=np.uint8)
    cv2.circle(back, (cx, cy), int(r_out), 0, -1)
    cv2.circle(back, (cx, cy), 40, 245, -1)
    cv2.rectangle(back, (nx - 8, ny - 8), (nx + 8, ny + 8), 245, -1)
    # 前光视图（缺陷用）
    front = np.full((300, 400), 60, dtype=np.uint8)
    cv2.circle(front, (cx, cy), int(r_out), 130, -1)
    cv2.circle(front, (cx, cy), 40, 40, -1)
    t = np.clip(rng.normal(0, 10, front.shape), -25, 25).astype(np.int16)
    front = np.clip(front.astype(np.int16) + t, 0, 255).astype(np.uint8)
    if has_defect:
        for _ in range(int(rng.integers(1, 3))):
            aa = rng.uniform(0, 2 * np.pi)
            rr = rng.integers(0, int(r_out) - 20)
            dx = int(cx + rr * np.cos(aa))
            dy = int(cy + rr * np.sin(aa))
            cv2.circle(front, (dx, dy), int(rng.integers(5, 12)), int(rng.integers(200, 240)), -1)
    # 真值
    d_true = 2 * r_out * MM_PER_PX
    size_ng = abs(d_true - REF_DIAM_MM) > TOL_MM
    truth = "NG" if (size_ng or has_defect) else "OK"
    return back, front, truth

# ---------- 3. 视觉处理函数 ----------
def locate(back):
    """模板匹配定位垫片中心"""
    res = cv2.matchTemplate(back, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return (max_loc[0] + 80, max_loc[1] + 80), max_val

def measure_diameter(back, center):
    """背光 ROI 内测直径（像素 -> 毫米）"""
    cx, cy = center
    roi = back[cy - 95:cy + 95, cx - 95:cx + 95]
    _, b = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_not(b)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = max(contours, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(big)
    return 2 * r * MM_PER_PX

def check_defect(front, center):
    """前光 ROI 内找异常亮斑，返回面积和是否超限"""
    cx, cy = center
    roi = front[cy - 100:cy + 100, cx - 100:cx + 100]
    _, b = cv2.threshold(roi, 190, 255, cv2.THRESH_BINARY)
    area = np.count_nonzero(b == 255)
    return area, area > DEFECT_LIMIT

# ---------- 4. 主循环：检测 100 个样品 ----------
samples = [make_sample() for _ in range(N_SAMPLES)]
os.makedirs("项目结果", exist_ok=True)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(PLC_ADDR)
print(f"已连接模拟PLC，开始检测 {N_SAMPLES} 个样品...")

t0 = time.time()
stats = {"correct": 0, "false_ng": 0, "false_ok": 0}
for i, (back, front, truth) in enumerate(samples, 1):
    center, score = locate(back)
    d_mm = measure_diameter(back, center)
    area, ng_defect = check_defect(front, center)
    ng_size = abs(d_mm - REF_DIAM_MM) > TOL_MM
    if ng_size and ng_defect:
        reason = "SIZE+DEFECT"
    elif ng_size:
        reason = "SIZE"
    elif ng_defect:
        reason = "DEFECT"
    else:
        reason = "CLEAN"
    verdict = "NG" if reason != "CLEAN" else "OK"
    sock.sendall((f"{verdict},{d_mm:.2f},{reason}\n").encode("utf-8"))

    if verdict == truth:
        stats["correct"] += 1
    elif verdict == "NG" and truth == "OK":
        stats["false_ng"] += 1
    else:
        stats["false_ok"] += 1

    # 保存标注结果图：左=背光(画测量圆)，右=前光(圈缺陷)
    back_c = cv2.cvtColor(back, cv2.COLOR_GRAY2BGR)
    front_c = cv2.cvtColor(front, cv2.COLOR_GRAY2BGR)
    cx, cy = center
    r_px = (d_mm / MM_PER_PX) / 2
    cv2.circle(back_c, (cx, cy), int(r_px), (0, 255, 0), 2)     # 绿圈 = 测量的直径
    cv2.circle(back_c, (cx, cy), 4, (0, 255, 0), -1)            # 绿点 = 中心
    roi_f = front[cy - 100:cy + 100, cx - 100:cx + 100]
    _, bf = cv2.threshold(roi_f, 190, 255, cv2.THRESH_BINARY)
    fc, _ = cv2.findContours(bf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cc in fc:
        if cv2.contourArea(cc) > 20:
            x2, y2, w2, h2 = cv2.boundingRect(cc)
            cv2.rectangle(front_c, (cx - 100 + x2, cy - 100 + y2),
                          (cx - 100 + x2 + w2, cy - 100 + y2 + h2), (0, 0, 255), 2)  # 红框 = 缺陷
    vis = np.hstack([back_c, front_c])
    color = (0, 0, 255) if verdict == "NG" else (0, 255, 0)
    cv2.putText(vis, f"VERDICT={verdict} DIAM={d_mm:.2f}mm DEFECT={area}px REASON={reason}", (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(vis, "RULE: 8.90<=D<=9.10 AND defect<=80px -> OK | GREEN=measured RED=defect", (8, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    imwrite_unicode(f"项目结果/样本{i:03d}.png", vis)
    if i % 10 == 0:
        print(f"已完成 {i}/{N_SAMPLES}")

elapsed = time.time() - t0
sock.close()

# ---------- 5. 验收统计 ----------
ok_truth = sum(1 for s in samples if s[2] == "OK")
ng_truth = N_SAMPLES - ok_truth
acc = stats["correct"] / N_SAMPLES * 100
print("========== 验收统计 ==========")
print(f"总样品: {N_SAMPLES}（真值 OK={ok_truth} NG={ng_truth}）")
print(f"正确率: {acc:.1f}%（要求 >= 95%）")
print(f"误杀率: 好品被判NG = {stats['false_ng']} 个")
print(f"漏检率: 坏品被判OK = {stats['false_ok']} 个")
print(f"节拍: {elapsed / N_SAMPLES * 1000:.0f} ms/件（要求 < 1000ms）")
print(f"标注结果图已保存到 项目结果 文件夹")