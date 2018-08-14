from utils import detector_utils as detector_utils
#from CNNGestureRecognizer import gestureClassifier as gClassifier
from CNNGestureRecognizer import gestureCNN as myNN
import cv2
import tensorflow as tf
import datetime
import argparse
import os

cap = cv2.VideoCapture(0)
detection_graph, sess = detector_utils.load_inference_graph()

# NN stuffs
import time
import numpy as np
mod = 0
lastgesture = -1

# Sample getter
numOfSamples = 2000
gestname = ""
path = ""
saveImg = False
counter = 0

real_punch = 0
esti_punch = 0

def saveROIImg(img):
    global counter, gestname, path, saveImg
    if counter > (numOfSamples - 1):
        # Reset the parameters
        saveImg = False
        gestname = ''
        counter = 0
        return
    
    counter = counter + 1
    name = gestname + str(counter)
    print("Saving img:",name)
    cv2.imwrite(path+name + ".png", img)
    time.sleep(0.04 )

def skinExtractor(roi):
    skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # HSV values
    low_range = np.array([0, 30, 60])
    upper_range = np.array([20, 200, 255])
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Apply skin color range
    mask = cv2.inRange(hsv, low_range, upper_range)
    mask2 = cv2.inRange(hsv,np.array([168,30,60]),np.array([180,200,255]))
    mask = cv2.bitwise_or(mask,mask2)
    cv2.imshow('after inRange',mask)
    mask = cv2.erode(mask, skinkernel, iterations=1)
    cv2.imshow('after erode',mask)
    mask = cv2.dilate(mask, skinkernel, iterations=1)
    cv2.imshow('after dilate',mask)
    # blur
    mask = cv2.GaussianBlur(mask, (15, 15), 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) 
    cv2.imshow('after blur',mask)
    #cv2.imshow("Blur", mask)
    # bitwise and mask original frame
    res = cv2.bitwise_and(roi, roi, mask=mask)
    # color to grayscale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    res = cv2.bilateralFilter(res,3,160,160)
    return res

def skinMask(frame, x0, y0, width, height):
    global visualize, mod, lastgesture, saveImg, esti_punch
    f_height = frame.shape[0]
    f_width = frame.shape[1]
    cv2.rectangle(frame, (x0, y0), (x0+width, y0+height), (0, 255, 0), 1)
    roi = frame[y0:y0+height, x0:x0+width]
    if(roi.shape[0] <= 0 or roi.shape[1] <= 0):
        return None
    # use black to fill in the rest of the part
    if(x0 < 0):
        roi = cv2.copyMakeBorder(
            roi, 0, 0, -x0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    if(y0 < 0):
        roi = cv2.copyMakeBorder(
            roi, -y0, 0, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    if(x0+width > f_width):
        roi = cv2.copyMakeBorder(
            roi, 0, 0, 0, x0+width-f_width, cv2.BORDER_CONSTANT, (0, 0, 0))
    if(y0+height > f_height):
        roi = cv2.copyMakeBorder(
            roi, 0, y0+height-f_height, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    # drop imgs that are too small
    # threshold = 20 %
    drop_thres = 5
    if(roi.shape[0]*100/frame.shape[0] < drop_thres):
        return None

    cv2.imshow('Colored ROI', roi)
    roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_CUBIC)

    res = skinExtractor(roi)

    retgesture = myNN.guessGesture(mod, res)
    if saveImg is True:
        saveROIImg(res)

    if lastgesture != retgesture:
        lastgesture = retgesture
        print myNN.output[lastgesture]

        if(lastgesture == 3):
            import subprocess
            subprocess.call(["xdotool", "type", ' '])
            print myNN.output[lastgesture] + "= Dino JUMP!"
	    esti_punch += 1
            time.sleep(0.01)

    return res


def Main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-sth', '--scorethreshold', dest='score_thresh', type=float,
                        default=0.48, help='Score threshold for displaying bounding boxes')
    parser.add_argument('-fps', '--fps', dest='fps', type=int,
                        default=1, help='Show FPS on detection/display visualization')
    parser.add_argument('-src', '--source', dest='video_source',
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=640, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=480, help='Height of the frames in the video stream.')
    parser.add_argument('-ds', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=16, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    args = parser.parse_args()

    global cap,saveImg,gestname,path, real_punch, esti_punch
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 1

    mod = myNN.loadCNN(0)
    while True:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, image_np = cap.read()
        frame = image_np
        # image_np = cv2.flip(image_np, 1)
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        # actual detection
        boxes, scores = detector_utils.detect_objects(
            image_np, detection_graph, sess)

        # draw bounding boxes
        detector_utils.draw_box_on_image(
            num_hands_detect, args.score_thresh, scores, boxes, im_width, im_height, image_np)

        # get boxes
        result = detector_utils.get_box_on_image(
            num_hands_detect, args.score_thresh, scores, boxes, im_width, im_height)
        for res in result:
            #img = frame[res['top']:res['bottom'],res['left']:res['right']]
            img = frame
            width = res['right']-res['left']
            height = res['bottom']-res['top']
            if(width <= 0 or height <= 0):
                continue
            ww = width if(width > height) else height
            # expands the ROI by 20%
            expansion = int(ww*0.2)
            masked = skinMask(img, res['left']-expansion, res['top']-expansion, ww+2*expansion, ww+2*expansion)
            if not masked is None:
                cv2.imshow('ROI', masked)

        # Calculate Frames per second (FPS)
        num_frames += 1
        elapsed_time = (datetime.datetime.now() -
                        start_time).total_seconds()
        fps = num_frames / elapsed_time

        if (args.display > 0):
            # Display FPS on frame
            if (args.fps > 0):
                detector_utils.draw_fps_on_image(
                    "FPS : " + str(int(fps)), image_np)

            cv2.imshow('Single Threaded Detection', cv2.cvtColor(
                image_np, cv2.COLOR_RGB2BGR))

            key = cv2.waitKey(25) & 0xFF 
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
            elif key == ord('s'):
                saveImg = not saveImg
                if gestname != '':
                    saveImg = True
                else:
                    print "Enter a gesture group name first, by pressing 'n'"
                    saveImg = False
            elif key == ord('n'):
                gestname = raw_input("Enter the gesture folder name: ")
                try:
                    os.makedirs(gestname)
                except OSError as e:
                    # if directory already present
                    if e.errno != 17:
                        print 'Some issue while creating the directory named -' + gestname
            
                path = "./"+gestname+"/"
	    elif key == ord(' '):
                real_punch += 1
		print "real_punch:",real_punch,"estimated_punch:",esti_punch
        else:
            print("frames processed: ",  num_frames,
                  "elapsed time: ", elapsed_time, "fps: ", str(int(fps)))


if __name__ == '__main__':
    Main()
