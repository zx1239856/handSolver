from utils import detector_utils as detector_utils
#from CNNGestureRecognizer import gestureClassifier as gClassifier
from CNNGestureRecognizer import gestureCNN as myNN
import cv2
import tensorflow as tf
import datetime
import argparse

cap = cv2.VideoCapture(0)
detection_graph, sess = detector_utils.load_inference_graph()

## NN stuffs
import time
import numpy as np
mod = 0
lastgesture = -1
kernel = np.ones((15,15),np.uint8)
kernel2 = np.ones((1,1),np.uint8)
skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

def skinMask(frame, x0, y0, width, height ):
    global guessGesture, visualize, mod, lastgesture, saveImg
    # HSV values
    low_range = np.array([0, 50, 80])
    upper_range = np.array([30, 200, 255])
    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]
    if(roi.shape[0]<=0 or roi.shape[1]<=0):
        return None
    cv2.imshow('ROI2', roi)
    roi = cv2.resize(roi,(200,200),interpolation=cv2.INTER_CUBIC)
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    #Apply skin color range
    mask = cv2.inRange(hsv, low_range, upper_range)
    
    mask = cv2.erode(mask, skinkernel, iterations = 1)
    mask = cv2.dilate(mask, skinkernel, iterations = 1)
    
    #blur
    mask = cv2.GaussianBlur(mask, (15,15), 1)
    #cv2.imshow("Blur", mask)
    
    #bitwise and mask original frame
    res = cv2.bitwise_and(roi, roi, mask = mask)
    # color to grayscale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    
    
        
    retgesture = myNN.guessGesture(mod, res)
    if lastgesture != retgesture :
        lastgesture = retgesture
        print myNN.output[lastgesture]
        
        if(lastgesture == 3):
            import subprocess
            subprocess.call(["xdotool", "type", ' '])
            print myNN.output[lastgesture] + "= Dino JUMP!"
            time.sleep(0.01)
    
    return res

def Main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-sth', '--scorethreshold', dest='score_thresh', type=float,
                        default=0.46, help='Score threshold for displaying bounding boxes')
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
                        default=8, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    args = parser.parse_args()

    global cap
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 2
    
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
        result = detector_utils.get_box_on_image(num_hands_detect, args.score_thresh, scores, boxes, im_width, im_height)
        for res in result:
            #img = frame[res['top']:res['bottom'],res['left']:res['right']]
            img = frame
            width = res['right']-res['left']
            height = res['bottom']-res['top']
            if(width<=0 or height<=0):
                continue
            ww = width if(width > height) else height
            masked = skinMask(img,res['left']-40,res['top']-40,ww+80,ww+80)
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

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            print("frames processed: ",  num_frames,
                  "elapsed time: ", elapsed_time, "fps: ", str(int(fps)))

if __name__ == '__main__':
    Main()
