#!/usr/bin/env python
from __future__ import print_function
import roslib
import sys
import rospy
import cv2
import numpy as np
import math

from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from std_msgs.msg import Int32
from std_msgs.msg import Float32
from std_msgs.msg import UInt16

from function import *
import numpy as np

#This part is for the camera calibration
mtx = np.array([[254.99142544,   0.,         311.43317687],
 [  0.,         253.087548,   254.01836666],
 [  0.,           0.,           1.,        ]])

dist = np.array([[-0.24186327,  0.03653006, -0.00916361, -0.00039632, -0.00048728]])


prev_command = 0



class core_processing:

    def __init__(self):
        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber("image_raw",Image,self.callback) #original image subscriber
        self.stepper = rospy.Publisher('stepper_control',Int32,queue_size=10)
        self.speed = rospy.Publisher('spd',Int32,queue_size=10)
        self.arm = rospy.Publisher('servo_control',Int32,queue_size=10)

    def callback(self,data):

        global prev_command
        try:    
            #image image to calibrated
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            cv_image = cv2.resize(cv_image,dsize=(640,480),interpolation=cv2.INTER_AREA)

            h,  w = cv_image.shape[:2]
            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
            dst = cv2.undistort(cv_image, mtx, dist, None, newcameramtx)
            x,y,w,h = roi
            dst = dst[y:y+h, x:x+w]
            cv_image = dst

            #set the region of interest
            ROI_image = cv_image[200:400, :]

            #set ROI image to image that will be processed
            cv_image = ROI_image

            #Get the height and width of the image
            height, width = cv_image.shape[:2]

            #Make image to gray scale
            gray_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            #Gaussian blur to eliminate noise of the image
            blur_img = gaussian_blur(gray_img, 5)

            #Canny conversion to get the line in the image
            canny_img = canny(blur_img, 50, 150)

            vertices = np.array([[(50,height),(width/2-45, height/2+60), (width/2+45, height/2+60), (width-50,height)]], dtype=np.int32)

            #get hough lines
            hough_img, lines = hough_lines(canny_img, 1, 1 * np.pi/180, 55, 10, 5)

            #Make grayscale image to BGR image
            canny_img = cv2.cvtColor(canny_img, cv2.COLOR_GRAY2BGR)

            #get the contact points
            points = getContactPoints(canny_img)

            #draw points
            canny_img = drawContactPoints(canny_img, points)

            #draw crossline
            canny_img = draw_crossLines(canny_img)

            #integrate two image(hough,canny)
            hough_img = weighted_img(hough_img, canny_img)

            H1LD = points['H1LD']
            H1RD = points['H1RD']
            H2LD = points['H2LD']
            H2RD = points['H2RD']
            H3LD = points['H3LD']
            H3RD = points['H3RD']
            H1Y = int(height/3)*1
            H2Y = int(height/3)*2
            H3Y = int(height/3)*3

            V1D = points['V1D']
            V2D = points['V2D']
            V3D = points['V3D']
            V4D = points['V4D']
            V5D = points['V5D']
            V6D = points['V6D']
            V7D = points['V7D']
            V1X = int(width/8)*1        # 1 block distance by y axis
            V2X = int(width/8)*2        # 2 blocks distance by y axis
            V3X = int(width/8)*3        # 3 blocks distance by y axis
            V4X = int(width/8)*4        # 4 blocks distance by y axis
            V5X = int(width/8)*5        # 5 blocks distance by y axis
            V6X = int(width/8)*6        # 6 blocks distance by y axis
            V7X = int(width/8)*7        # 7 blocks distance by y axis

            XD1 = int(width/8)*1        # 1 block distance by x axis
            XD2 = int(width/8)*2        # 2 blocks distance by x axis
            XD3 = int(width/8)*3        # 3 blocks distance by x axis

            STOP = 12
            STRAIGHT = 0
            TO_3 = 3
            BACKWARD = 6
            TO_9 = 9
            CLOCKWISE = 33
            COUNTER_CLOCKWISE = 99
            RIGHT_1 = 110
            RIGHT_2 = 120
            RIGHT_3 = 130
            LEFT_1 = 210
            LEFT_2 = 220
            LEFT_3 = 230
            HLD = [H1LD,H2LD,H3LD]
            HRD = [H1RD,H2RD,H3RD]
            VD = [V1D,V2D,V3D,V4D,V5D,V6D,V7D]
            PRO_VD = [i for i in VD if i != 0]
            X_diff = (sum(HLD)-sum(HRD))/3
            X_mean = HLD[0]+HRD[0]/2
            PROY_mean = sum(PRO_VD)/len(PRO_VD)
            Y_mean = sum(VD)/len(VD)
            leftIncline = (VD[0]*6 + 3*VD[1] + 2*VD[2]) / 480
            rightIncline = (VD[4]*2 + 3*VD[5] + 6*VD[6]) / 480
            
            leftLane, rightLane, endLane = getLane(lines) # determine the existance of the line(left,right,end)
            left = right = end = False
            if len(leftLane) != 0:
                left = True
            if len(rightLane) != 0:
                right = True
            if len(endLane) != 0:
                end = True
            
            ############################### algorithm ######################################################################
            speed_command = STRAIGHT
            if X_diff < -10:
                speed_command = RIGHT_1
            elif X_diff > 20:
                speed_command = LEFT_1
            if Y_mean > 130:
                if sum(VD[:3]) > sum(VD[4:]):
                    speed_command = LEFT_2
                else:
                    speed_command = RIGHT_2
            
            if Y_mean < 80:
                speed_command = BACKWARD
            print("====Values====")
            print("Y_mean :",Y_mean)
            print("X_mean :",X_mean)
            print("X_diff",X_diff)
            print("leftIncline :",leftIncline)
            print("rightIncline :",rightIncline)
<<<<<<< HEAD
=======

>>>>>>> 8d838bcd73b173009bf8bd148747a70aae345154
            ############################### algorithm ######################################################################

            
            
            self.stepper.publish(speed_command)

            #speed could be 122,490,3921
            #self.speed.publish(490)

            cv2.imshow('hough',hough_img)
            cv2.imshow('cv',cv_image)
            #cv2.imshow('ROI',ROI_image)

            cv2.waitKey(1)



        except CvBridgeError as e:
            print(e)

def main(args):
    cp = core_processing()
    rospy.init_node('core', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
   
if __name__ == '__main__':
    main(sys.argv)