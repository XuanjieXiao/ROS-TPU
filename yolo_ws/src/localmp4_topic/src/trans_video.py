#!/usr/bin/env python
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def publish_video(video_path):
    # 初始化ROS节点
    rospy.init_node('video_publisher', anonymous=True)
    # 创建一个Publisher，发布到'/camera/image'话题，消息类型为Image
    pub = rospy.Publisher('/camera/image', Image, queue_size=10)
    # 设置帧率
    rate = rospy.Rate(30) # 30hz

    # 使用OpenCV读取视频
    cap = cv2.VideoCapture(video_path)
    # 创建CvBridge对象
    bridge = CvBridge()

    while not rospy.is_shutdown() and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            try:
                # 将OpenCV的图像转换为ROS的Image消息
                ros_image = bridge.cv2_to_imgmsg(frame, "bgr8")
                # 发布图像
                pub.publish(ros_image)
                print("converting ....... &&  publishing images")
                rate.sleep()
            except CvBridgeError as e:
                print(e)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 如果视频播放完毕，重新开始

    cap.release()

if __name__ == '__main__':
    try:
        # 视频文件路径
        video_path = "./datas/datasets/test_car_person_1080P.mp4"
        publish_video(video_path)
    except rospy.ROSInterruptException:
        pass
