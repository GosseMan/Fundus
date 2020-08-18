import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import numpy as np
import imageio
import os
from glob import glob

def zoomin(in_path, out_path, zoomin):
    seq = iaa.Sequential([
    iaa.Affine(zoomin)
])
    imglist = []
    img = cv2.imread(in_path)
    imglist.append(img)
    imgname = in_path.split('/')[-1].split('.')[0]
    ex = in_path.split('/')[-1].split('.')[1]
    img_zoom = seq.augment_images(imglist)
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    cv2.imwrite(out_path+'/'+imgname+'_zoomin_'+str(zoomin)+'.'+ex,img_zoom[0])

def zoomin_dir(in_path, out_path, zoomin):
    seq = iaa.Sequential([
    iaa.Affine(zoomin)
])
    files = glob(in_path+'/' + '*.jpg') # jpg or png or tif etc..
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    for file in files:
        imglist = []
        img = cv2.imread(file)
        imglist.append(img)
        imgname = file.split('/')[-1].split('.')[0]
        ex = file.split('/')[-1].split('.')[1]
        img_zoom = seq.augment_images(imglist)
        cv2.imwrite(out_path+'/'+imgname+'_zoomin_'+str(zoomin)+'.'+ex, img_zoom[0])

def zoomin_cut(in_path, out_path, zoomin):
    seq = iaa.Sequential([
    iaa.Affine(zoomin)
])
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    for file in files:
        imglist = []
        img = cv2.imread(file)
        imglist.append(img)
        imgname = in_path.split('/')[-1].split('.')[0]
        ex = in_path.split('/')[-1].split('.')[1]
        img_zoom = seq.augment_images(imglist)
        image_zoomed = img_zoom[0]
        origin = cv2.imread(in_path,cv2.IMREAD_GRAYSCALE)
        ret, circle = cv2.threshold(origin, 10, 255, cv2.THRESH_BINARY)
        for x in range(img_color.shape[0]):
            for y in range(img_color.shape[1]):
                if circle[x][y] == 0:
                    for z in range(img_color.shape[2]):
                        image_zoomed[x][y][z] = 0
    cv2.imwrite(out_path+'/'+imgname+'_zoomin_'+str(zoomin)+'.'+ex, image_zoomed)

def zoomin_cut_dir(in_path, out_path, zoomin):
    seq = iaa.Sequential([
    iaa.Affine(zoomin)
])
    files = glob(in_path+'/' + '*.jpg')
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    for file in files:
        imglist = []
        img = cv2.imread(file)
        imglist.append(img)
        imgname = file.split('/')[-1].split('.')[0]
        ex = file.split('/')[-1].split('.')[1]
        img_zoom = seq.augment_images(imglist)
        image_zoomed = img_zoom[0]
        origin = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
        ret, circle = cv2.threshold(origin, 1, 255, cv2.THRESH_BINARY)
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if circle[x][y] == 0:
                    for z in range(img.shape[2]):
                        image_zoomed[x][y][z] = 0
        cv2.imwrite(out_path+'/'+imgname+'_filter_'+str(zoomin)+'.'+ex, circle)
        cv2.imwrite(out_path+'/'+imgname+'_zoomin_'+str(zoomin)+'.'+ex, image_zoomed)

def zoom_use_filter(in_path, out_path, zoomin,filter):
    seq = iaa.Sequential([
    iaa.Affine(zoomin)
])
    files = glob(in_path+'/' + '*.jpg')
    zoom_rate = str(round((zoomin*100)-100,0))
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    for file in files:
        imglist = []
        img = cv2.imread(file)
        imglist.append(img)
        imgname = file.split('/')[-1].split('.')[0]
        ex = file.split('/')[-1].split('.')[1]
        img_zoom = seq.augment_images(imglist)
        image_zoomed = img_zoom[0]
        h, w, c = img.shape

        '''
        if w == 1920:
            filter = cv2.imread('/home/psj/Desktop/psj_util/1920X1296/average_binary.jpg')
        elif w == 3872:
            filter = cv2.imread('/home/psj/Desktop/psj_util/3872X2592/average_binary.jpg')
        else:
            print('Zoomin Error : Figure Size Error')
            return
        '''

        #filter = filter/255
        final = np.multiply(image_zoomed, filter)
        #cv2.imwrite(out_path+'/'+imgname+'_zoomin_origin_'+str(zoomin)+'.'+ex, image_zoomed)
        cv2.imwrite(out_path+'/'+imgname+'_zoom'+zoom_rate+'.'+ex, final)

def main():
    PATH = '/home/psj/Desktop/exam'
    OUTPUT_PATH = '/home/psj/Desktop/exam_1'
    filter = cv2.imread(('/home/psj/Desktop/psj_util/560X480/threshold40.jpg'))
    filter=filter/255
    zoomin_dir(PATH,OUTPUT_PATH, 1.2)
    #zoomin_cut_dir(PATH,OUTPUT_PATH, 1.2)
    zoom_use_filter(PATH,OUTPUT_PATH,1.2,filter)

if __name__ == "__main__":
    main()
