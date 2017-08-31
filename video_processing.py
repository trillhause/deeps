import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')

import numpy as np
import cv2


def capture_frames(file_location):

	cap = cv2.VideoCapture(file_location)
	success, image = cap.read()
	count = 0
	success = True

	while success:
		success, image = cap.read()
		print 'Read a new frame: ', success
		cv2.imwrite("data/test_frames/frame%d.jpg" % count, image)
		cv2.imwrite("data/test_frames/frame%d.jpg" % count+1, image)
		count+=2
	
capture_frames('data/test.mp4')