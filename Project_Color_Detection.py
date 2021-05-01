import numpy as np
import argparse
import imutils
import cv2 as cv

#Directly specify the path to the video
parser = argparse.ArgumentParser()

parser.add_argument("-v", "--video", help="path to the video file")
parser.add_argument("-s", "--size", type=int, default=64)
args = vars(parser.parse_args())

#Set range for  colors
lower_colors = {'Black': (0, 0, 0), 'White': (0, 0, 128),'Red': (155, 25, 0), 'Green': (25, 52, 72),
                'Blue': (94, 80, 2), 'Yellow': (23, 59, 119),}
upper_colors = { 'Black': (50, 55, 55), 'White': (250, 250, 250),'Red': (179, 255, 255), 'Green': (102, 255, 255),
                 'Blue': (120, 255, 255), 'Yellow': (54, 255, 255),}

colors = {'Black': (170, 150, 50), 'White': (255, 255, 255),'Red': (0, 0, 255), 'Green': (0, 255, 0),
          'Blue': (255, 0, 0), 'Yellow': (0, 255, 217),}

#take a link to a webcam
if not args.get("v", False):
    camera = cv.VideoCapture(0)

#take the link to the video file
else:
    camera = cv.VideoCapture(args["v"])

#loop over frames from the video file stream
while True:

#grab the frame from the threaded video file stream
    (grabbed, frame) = camera.read()
    if args.get("v") and not grabbed:
        break

    # resize frame
    frame = imutils.resize(frame, width=1000)

    blurred = cv.GaussianBlur(frame, (11, 11), 0)
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

    # construct a mask for the colors
    for key, value in upper_colors.items():

        kernel = np.ones((7, 7), np.uint8)
#masking the image to find our color
        mask = cv.inRange(hsv, lower_colors[key], upper_colors[key])
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

        # find contours
        cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
                                cv.CHAIN_APPROX_SIMPLE)[-2]
        center = None

        if len(cnts) > 0:

            c = max(cnts, key=cv.contourArea)
            ((x, y), radius) = cv.minEnclosingCircle(c)
            M = cv.moments(c)
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])

            rect = cv.minAreaRect(c)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(frame,[box],0,(0,0,0),2)

            if radius > 0.5:
                cv.circle(frame,(cx,cy) , int(radius), colors[key], 5)
                cv.putText(frame, key + "color",(cx-15,cy-15) , cv.FONT_HERSHEY_SIMPLEX, 0.6,
                            colors[key], 2)

    # show the frame to our screen
    cv.imshow("Frame", frame)

    k = cv.waitKey(5)
    if k == 27:
        break

#release the captured frame
cap.release()
#destroys all windows
cv.destroyAllWindows()
