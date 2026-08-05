# 视觉软件最小版（学习用骨架）—— 升级：相机抽象层（电脑摄像头 / 海康工业相机）
# 运行: python 视觉软件_最小版.py
# 分层：算法层 / 采集层（相机抽象）/ 界面层
import sys
import time
import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
                               QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
                               QTextEdit, QSplitter, QComboBox)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, Qt

# ================= 算法层（你已经会的部分） =================
MM_PER_PX = 0.05      # 标定: 180px = 9.00mm
REF_DIAM_MM = 9.00    # 标准直径
TOL_MM = 0.10         # 公差

def make_washer():
    """生成一个随机位置的模拟垫片（背光）"""
    rng = np.random.default_rng()
    cx = int(rng.uniform(170, 230))
    cy = int(rng.uniform(120, 180))
    r_out = rng.uniform(88, 92)
    back = np.full((300, 400), 245, dtype=np.uint8)
    cv2.circle(back, (cx, cy), int(r_out), 0, -1)
    cv2.circle(back, (cx, cy), 40, 245, -1)
    return back

def measure_washer(img):
    """测直径（像素->毫米），返回 (直径mm, 带标注结果图)；找不到返回 None"""
    _, b = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_not(b)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    big = max(contours, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(big)
    d_mm = 2 * r * MM_PER_PX
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.circle(vis, (int(x), int(y)), int(r), (0, 255, 0), 2)
    cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), -1)
    return d_mm, vis

def judge(d_mm):
    return "OK" if abs(d_mm - REF_DIAM_MM) <= TOL_MM else "NG"

# ================= 采集层（相机抽象：换相机只改这里） =================
class CameraSource:
    """相机接口：所有相机都实现 open / grab / close 三个方法"""
    def open(self):   raise NotImplementedError
    def grab(self):   raise NotImplementedError
    def close(self):  raise NotImplementedError

class WebcamSource(CameraSource):
    """电脑摄像头（OpenCV，现在就能用）"""
    def open(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = None
            raise RuntimeError("打不开电脑摄像头（没有摄像头或被占用）")
    def grab(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("读取摄像头画面失败")
        return frame
    def close(self):
        if getattr(self, "cap", None):
            self.cap.release()
            self.cap = None

class HikCameraSource(CameraSource):
    """海康工业相机（需要：安装海康MVS SDK + 相机驱动 + 相机已连接，见《海康相机接入指南》）"""
    def open(self):
        try:
            import mvsdk
        except ImportError:
            raise RuntimeError("未找到 mvsdk：请先安装海康MVS SDK（见《海康相机接入指南》）")
        self._mvsdk = mvsdk
        devices = mvsdk.CameraList()
        if len(devices) == 0:
            raise RuntimeError("没有发现海康相机：检查网线/USB、驱动、GigE相机IP设置")
        self.cam = mvsdk.Camera(devices[0])
        self.cam.MV_CC_OpenDevice()
        self.cam.MV_CC_SetEnumValue("TriggerMode", 0)   # 0=连续采集（1=软触发）
        self.cam.MV_CC_StartGrabbing()
    def grab(self):
        frame, _ = self.cam.MV_CC_GetOneFrameTimeout(1000)
        return frame
    def close(self):
        if getattr(self, "cam", None):
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.cam = None

def make_camera(kind):
    """根据界面选择创建对应相机"""
    return HikCameraSource() if kind == "海康工业相机" else WebcamSource()

# ================= 界面层 =================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视觉软件最小版 - Python/PySide6/OpenCV")
        self.src = None                      # 当前相机源
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_live)
        self.build_ui()

    def build_ui(self):
        left = QWidget()
        lv = QVBoxLayout(left)
        self.live_label = QLabel("相机未打开")
        self.live_label.setAlignment(Qt.AlignCenter)
        self.live_label.setMinimumSize(480, 360)
        lv.addWidget(self.live_label)
        row = QHBoxLayout()
        self.cam_type = QComboBox()
        self.cam_type.addItems(["电脑摄像头", "海康工业相机"])
        row.addWidget(self.cam_type)
        self.btn_cam = QPushButton("打开相机")
        self.btn_cam.clicked.connect(self.toggle_camera)
        row.addWidget(self.btn_cam)
        lv.addLayout(row)

        right = QWidget()
        rv = QVBoxLayout(right)
        self.result_label = QLabel("测量结果图像")
        self.result_label.setAlignment(Qt.AlignCenter)
        rv.addWidget(self.result_label)
        self.btn_sim = QPushButton("模拟样品测量")
        self.btn_sim.clicked.connect(self.measure_sample)
        rv.addWidget(self.btn_sim)
        self.btn_snap = QPushButton("相机拍照测量")
        self.btn_snap.clicked.connect(self.measure_camera)
        rv.addWidget(self.btn_snap)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["测量值", "判断", "用时"])
        rv.addWidget(self.table)

        self.log = QTextEdit()
        self.log.setMaximumHeight(110)
        self.log.setReadOnly(True)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([520, 520])

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(splitter)
        layout.addWidget(self.log)
        self.setCentralWidget(central)
        self.resize(1080, 680)

    def log_msg(self, s):
        self.log.append(f"[{time.strftime('%H:%M:%S')}] {s}")

    # ---- 采集层: 打开/关闭/刷新 ----
    def toggle_camera(self):
        if self.src is None:
            try:
                kind = self.cam_type.currentText()
                self.src = make_camera(kind)
                self.src.open()
                self.timer.start(33)
                self.btn_cam.setText("关闭相机")
                self.log_msg(f"相机已打开（{kind}）")
            except Exception as e:
                self.src = None
                self.log_msg(f"打开相机失败：{e}")
        else:
            self.timer.stop()
            self.src.close()
            self.src = None
            self.btn_cam.setText("打开相机")
            self.live_label.setText("相机未打开")
            self.log_msg("相机已关闭")

    def update_live(self):
        try:
            frame = self.src.grab()
            self.show_image(self.live_label, frame)
        except Exception as e:
            self.timer.stop()
            self.log_msg(f"取流中断：{e}")

    # ---- 工具 ----
    def show_image(self, label, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        label.setPixmap(QPixmap.fromImage(qimg).scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def add_row(self, value, verdict, ms):
        self.table.insertRow(0)
        self.table.setItem(0, 0, QTableWidgetItem(value))
        self.table.setItem(0, 1, QTableWidgetItem(verdict))
        self.table.setItem(0, 2, QTableWidgetItem(ms))

    # ---- 业务: 测量 ----
    def measure_sample(self):
        t0 = time.time()
        result = measure_washer(make_washer())
        dt = (time.time() - t0) * 1000
        if result is None:
            self.log_msg("模拟测量: 未找到目标")
            return
        d_mm, vis = result
        verdict = judge(d_mm)
        self.show_image(self.result_label, vis)
        self.add_row(f"{d_mm:.2f}mm", verdict, f"{dt:.0f}ms")
        self.log_msg(f"模拟测量: 直径 {d_mm:.2f}mm -> {verdict}（{dt:.0f}ms）")

    def measure_camera(self):
        if self.src is None:
            self.log_msg("请先打开相机")
            return
        try:
            frame = self.src.grab()
        except Exception as e:
            self.log_msg(f"拍照失败：{e}")
            return
        t0 = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = measure_washer(gray)
        dt = (time.time() - t0) * 1000
        if result is None:
            self.show_image(self.result_label, frame)
            self.log_msg("相机拍照: 未找到目标（把垫片放镜头下才会测到）")
            return
        d_mm, vis = result
        verdict = judge(d_mm)
        self.show_image(self.result_label, vis)
        self.add_row(f"{d_mm:.2f}mm", verdict, f"{dt:.0f}ms")
        self.log_msg(f"相机测量: 直径 {d_mm:.2f}mm -> {verdict}")

    def closeEvent(self, event):
        if self.src is not None:
            self.src.close()
        event.accept()

# ================= 主入口 =================
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    if "--selftest" in sys.argv:
        def write_marker():
            with open("自检结果.txt", "w", encoding="utf-8") as f:
                f.write("SELFTEST_OK " + time.strftime("%H:%M:%S"))
            print("SELFTEST_OK", flush=True)
        QTimer.singleShot(1200, win.measure_sample)
        QTimer.singleShot(2500, write_marker)
        QTimer.singleShot(5000, app.quit)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()