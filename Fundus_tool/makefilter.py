import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import numpy as np
import imageio
import os
from glob import glob

def filter_dir(in_path, out_path):
    files = glob(in_path+'/' + '*.jpg')
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    for file in files:
        imgname = file.split('/')[-1].split('.')[0]
        origin = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
        h, w = origin.shape
        out = out_path+'/'+str(w)+'X'+str(h)
        if not os.path.isdir(out):
            os.mkdir(out)
        ret, circle = cv2.threshold(origin, 3, 255, cv2.THRESH_BINARY)
        cv2.imwrite(out+'/'+imgname+'_filter_.jpg', circle)

def filter_binary_average(in_path, out_path):
    files = glob(in_path+'/' + '*.jpg')
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    firstimg = cv2.imread(files[0],cv2.IMREAD_GRAYSCALE)
    h, w = firstimg.shape
    final = np.zeros((h,w))
    for file in files:
        imgname = file.split('/')[-1].split('.')[0]
        origin = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
        h, w = origin.shape
        out = out_path+'/'+str(w)+'X'+str(h)
        if not os.path.isdir(out):
            os.mkdir(out)
        ret, circle = cv2.threshold(origin, 2, 255, cv2.THRESH_BINARY)
        final = final + circle
        cv2.imwrite(out+'/'+imgname+'.jpg', circle)
    final = final/len(files)
    final = final.astype('uint8')

    cv2.imwrite(out+'/'+'filter.jpg', final)

def filter_average_binary(in_path, out_path):
    files = glob(in_path+'/' + '*.jpg')
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    count = 0
    firstimg = cv2.imread(files[0],cv2.IMREAD_GRAYSCALE)
    h, w = firstimg.shape
    out = out_path+'/'+str(w)+'X'+str(h)
    if not os.path.isdir(out):
        os.mkdir(out)
    final = np.zeros((h,w))
    for file in files:
        img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
        final = final + img
    final = final/len(files)
    #final = final.astype('uint8')
    cv2.imwrite(out+'/'+'average.jpg', final)
    thresholds = list(range(40,100,10))
    for threshold in thresholds:
        ret, circle = cv2.threshold(final, threshold, 255, cv2.THRESH_BINARY)
        cv2.imwrite(out+'/'+'threshold'+str(threshold)+'.jpg', circle)

def main():
    PATH = '/home/psj/Desktop/small'
    OUTPUT_PATH = '/home/psj/Desktop/small_filter'

    filter_average_binary(PATH,OUTPUT_PATH)

if __name__ == "__main__":
    main()
