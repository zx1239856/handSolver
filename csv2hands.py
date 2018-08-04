import cv2
import argparse
import numpy as np
import os
import csv

def findIndex(table,itemName):
    for index, item in enumerate(table):
        if(itemName == item):
            return index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', dest='source', type=str,
                        default='', help='Source path for imgs to crop')
    parser.add_argument('-d', '--dest', dest='destination', type=str,
                        default='', help='Dest path for imgs to save')
    parser.add_argument('-c','--csv',dest='csvFile',type=str,default='',help='CSV file path')
    args = parser.parse_args()
    dst = args.destination
    if(args.source == ''):
        print("Source folder not specified!\n")
        return
    elif(args.csvFile == ''):
        print("CSV file not specified!\n")
        return
    src = args.source.rstrip('/')
    if(dst==''):
        dst = src + "_output"
    print("===== Attempting to process imgs in folder ======\n")
    csv_file = csv.reader(open(args.csvFile,'r'))
    xminIdx = xmaxIdx = yminIdx = ymaxIdx = filenameIdx = 0
    lastfile = ""
    counter = 0
    xmin = xmax = ymin = ymax = 0

    if not os.path.exists(dst):
        os.makedirs(dst)

    for idx,line in enumerate(csv_file):
        if(idx == 0):
            xminIdx = findIndex(line,'xmin')
            xmaxIdx = findIndex(line,'xmax')
            yminIdx = findIndex(line,'ymin')
            ymaxIdx = findIndex(line,'ymax')
            filenameIdx = findIndex(line,'filename')
        else:
            ymin = int(line[yminIdx])
            ymax = int(line[ymaxIdx])
            xmin = int(line[xminIdx])
            xmax = int(line[xmaxIdx])
            try:
                img = cv2.imread(src + '/' + line[filenameIdx])
            except cv2.error as e:
                print("File "+line[filenameIdx]+" open failed, ignoring it...\n")
                continue
            print("Processing file: " + line[filenameIdx] + '\n')
            if(line[filenameIdx] != lastfile):
                cv2.imwrite(dst + '/' + line[filenameIdx],img[ymin:ymax,xmin:xmax])
                counter = 0
            else:
                cv2.imwrite(dst + '/' + os.path.splitext(line[filenameIdx])[0] + '_' + counter + os.path.splitext(line[filenameIdx])[1],
                img[ymin:ymax,xmin:xmax])
                counter = counter + 1
    print("Process complete!\n")
            
    

if __name__ == "__main__":
    main()