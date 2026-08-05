# 第3课 练习3：用摄像头采集图像
# 按键：空格 = 拍照保存，q = 退出
import cv2

def imwrite_unicode(path, img):
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(path)
    return ok

# 打开第一个摄像头（工业上这一步换成"连接工业相机SDK"）
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("打不开摄像头：可能没有摄像头，或被其他软件占用")
    exit()

# 查看相机的当前参数（工业相机上对应 SDK 里的配置项）
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"相机分辨率: {int(w)} x {int(h)}，帧率: {fps}")

# 尝试手动控制参数（很多电脑摄像头不响应，工业相机才可以完全控制）
cap.set(cv2.CAP_PROP_EXPOSURE, -5)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)
cap.set(cv2.CAP_PROP_CONTRAST, 110)

print("实时画面已打开：空格=拍照，q=退出")
while True:
    ok, frame = cap.read()
    if not ok:
        print("读取画面失败")
        break
    cv2.imshow("摄像头实时画面（空格拍照 q退出）", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):          # 空格：拍一张
        imwrite_unicode("练习3_拍照.png", frame)
        print("已保存照片：练习3_拍照.png")
    elif key == ord('q'):        # q：退出
        break

cap.release()
cv2.destroyAllWindows()
print("练习3结束")