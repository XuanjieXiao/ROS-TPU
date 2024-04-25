#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
class ImageSaver:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('image_saver', anonymous=True)
        # 创建CvBridge对象
        self.bridge = CvBridge()
        # 创建一个Subscriber，订阅'/camera/image'话题
        self.image_sub = rospy.Subscriber('/camera/image', Image, self.callback)
        # 设置保存图片的计数器
        self.image_count = 0

    def callback(self, data):
        try:
            # 将ROS的Image消息转换为OpenCV的图像格式
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # 保存图像
        output_dir = "./save_images"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        img_name = "./save_images/camera_image_{:04d}.jpeg".format(self.image_count)
        cv2.imwrite(img_name, cv_image)
        rospy.loginfo("Image saved as {}".format(img_name))
        self.image_count += 1

if __name__ == '__main__':
    image_saver = ImageSaver()
    try:
        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
