# 第1课 练习1：读图 → 灰度 → 观察
# 运行方法：在终端执行  python 练习1_读图与灰度.py
# 如果弹出窗口，按任意键关闭

import cv2
import numpy as np


# 现场实用技巧：OpenCV 的 imwrite/imread 对中文路径支持不好，
# 用下面这两个函数读写就永远不会乱码（工厂里文件夹常带中文）
def imwrite_unicode(path, img):
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(path)
    return ok


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)


# 1. 生成一张"模拟工件"图像并保存（先把读图的素材准备好）
img = np.full((400, 600), 200, dtype=np.uint8)      # 灰色底板，模拟金属
cv2.rectangle(img, (150, 100), (450, 300), 0, -1)   # 黑色"工件"矩形
cv2.circle(img, (300, 200), 50, 200, -1)            # 中间的"圆孔"
imwrite_unicode("工件_模拟图.png", img)
print("已生成并保存：工件_模拟图.png")

# 2. 读取图片（工业上就是从相机拿到这一张图）
image = imread_unicode("工件_模拟图.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. 查看基本信息
h, w = gray.shape
print(f"图像尺寸：宽 {w} 像素 x 高 {h} 像素")

# 重要：OpenCV 用 (x, y) 坐标（x 是列，y 是行），
#       numpy 数组用 [行, 列] 下标，也就是 gray[y, x]，顺序别搞反！
print(f"底板灰度值 gray[50, 50]   ：{gray[50, 50]}")
print(f"工件灰度值 gray[200, 200] ：{gray[200, 200]}")
print(f"圆孔中心 gray[200, 300]   ：{gray[200, 300]}")

# 4. 显示（按任意键关闭窗口）
cv2.imshow("原图", image)
cv2.imshow("灰度图", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()