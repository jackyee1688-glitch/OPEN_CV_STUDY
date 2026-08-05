# 演示版（与练习1逻辑相同，窗口8秒后自动关闭）
import cv2
import numpy as np

def imwrite_unicode(path, img):
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(path)
    return ok

def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)

# 第1步：生成模拟工件图
img = np.full((400, 600), 200, dtype=np.uint8)
cv2.rectangle(img, (150, 100), (450, 300), 0, -1)
cv2.circle(img, (300, 200), 50, 200, -1)
imwrite_unicode("工件_模拟图.png", img)
print("第1步：已生成并保存 工件_模拟图.png")

# 第2步：读取 + 转灰度
image = imread_unicode("工件_模拟图.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
print(f"第2步：读取图片成功，尺寸 宽{w} x 高{h}")

# 第3步：查看灰度值
print(f"第3步：底板灰度={gray[50,50]}，工件灰度={gray[200,200]}，圆孔中心灰度={gray[200,300]}")

# 生成一张拼接预览图（左边原图，右边灰度图），保存到工作区
gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
preview = np.hstack([image, gray_bgr])
imwrite_unicode("演示_原图与灰度图.png", preview)

# 第4步：弹出窗口显示，8秒后自动关闭
print("第4步：弹出窗口【原图】【灰度图】，8秒后自动关闭……")
cv2.imshow("原图", image)
cv2.imshow("灰度图", gray)
cv2.waitKey(8000)
cv2.destroyAllWindows()
print("演示结束")