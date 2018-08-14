"""
	A simple tool to calculate box range for dataset http://www.robots.ox.ac.uk/~vgg/data/hands/
	and convert the information to CSV file
	Licensed under MIT license
	Copyright: zx1239856@gmail.com     ZHANG XIANG
	## usage: use -s or --src param to specify input dir
"""

import cv2
import argparse
import numpy as np
import scipy.io as sio
import os
import csv

def save_csv(csv_path, csv_content):
    with open(csv_path, 'w') as csvfile:
        wr = csv.writer(csvfile)
        for i in range(len(csv_content)):
            wr.writerow(csv_content[i])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', dest='source', type=str,
                        default='', help='Source path for imgs to crop')
    parser.add_argument('-d', '--dest', dest='destination', type=str,
                        default='', help='Dest path for imgs to save')
    args = parser.parse_args()
    dst = args.destination
    if(args.source == ''):
        print("Source folder not specified!\n")
        return
    src = args.source.rstrip('/')
    if(dst==''):
        dst = src + "_output"
    print("===== Attempting to process imgs in folder ======\n")
    xmin = xmax = ymin = ymax = int(0)

    header = ['filename', 'width', 'height',
              'class', 'xmin', 'ymin', 'xmax', 'ymax']
    csvholder = []
    csvholder.append(header)

    if not os.path.exists(dst):
        os.makedirs(dst)
    for f in os.listdir(src + '/images/'):
        try:
            img = cv2.imread(src + '/images/' + f)
        except cv2.error as e:
            print("Error open this image, ignoring it...\n")
            continue
        # load mat file
        if(f.split('.')[0]==''):
            continue
        boxes = sio.loadmat(src + '/annotations/' + f.split('.')[0]+".mat")
        sp = img.shape
        counter = 0
	print("Process " + f + ", please wait","width:",sp[1],"height:",sp[0])
        for point in boxes['boxes'][0]:
            x = np.array([point[0][0][0][0][0],point[0][0][1][0][0],point[0][0][2][0][0],point[0][0][3][0][0]])
            y = np.array([point[0][0][0][0][1],point[0][0][1][0][1],point[0][0][2][0][1],point[0][0][3][0][1]])
            #print(point)
            xmin = int(np.min(x))
            xmax = int(np.max(x))
            ymin = int(np.min(y))
            ymax = int(np.max(y))
            if(xmin<=0):
                xmin=1
            if(xmax>=sp[0]):
                xmax=sp[0]
            if(ymin<=0):
                ymin=1
            if(ymax>=sp[1]):
                ymax=sp[1]
            #threshold = 20
            if(xmax-xmin <=0 or ymax-ymin <= 0):
                continue
            #cv2.imwrite(dst +'/' + f.split('.')[0] + '_' + bytes(counter) + '.' + f.split('.')[1],img[xmin:xmax,ymin:ymax])
	    #cv2.imshow('preview', img[xmin:xmax,ymin:ymax])
            #cv2.waitKey(3)
            # remember to exchange x & y here
            labelrow = [f,np.size(img, 1), np.size(img, 0), "hand", ymin, xmin, ymax, xmax]
            csvholder.append(labelrow)
        #counter +=1
        '''
        labelrow = [f,np.size(img, 1), np.size(img, 0), "hand", 1, 1, np.size(img, 1),np.size(img, 0)]
        csvholder.append(labelrow)
	'''
        #print(counter)
        #cv2.imwrite(dst +'/' + f,img[ymin:ymax,xmin:xmax])
    save_csv("result.csv", csvholder)
    print("Process complete!\n")
    cv2.destroyAllWindows()   
    

if __name__ == "__main__":
    main()
