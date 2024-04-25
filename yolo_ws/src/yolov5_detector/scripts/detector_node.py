#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from yolov5_opencv import detect_ros, argsparser, prepare_model




def detector():
    rospy.init_node('detector_node', anonymous=True)
    pub = rospy.Publisher('detection_results', String, queue_size=10)
    rate = rospy.Rate(30)  # 10hz

    args = argsparser()
    # 可以在这里修改 args 的默认值，或者从ROS参数服务器获取
    # 例如，args.input = rospy.get_param("~input_path", "../datasets/test")
    yolov5 = prepare_model(args)
    print('ok')
    
    while not rospy.is_shutdown():
        results = detect_ros(args, yolov5)
        rospy.loginfo(results)
        pub.publish(str(results))
        rate.sleep()

if __name__ == '__main__':
    try:
        detector()
    except rospy.ROSInterruptException:
        pass
