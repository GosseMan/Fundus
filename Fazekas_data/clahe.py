import argparse
import cv2
import numpy as np
import torch
import os
from torch.autograd import Function
from torchvision import models

def clahe(image_path,gridsize):
    bgr = cv2.imread(image_path)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

def clahe_allfiles_dir(src, OUTPUT_DIR, gridsize):
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    file_list = os.listdir(src)
    for name in file_list:
        bgr = cv2.imread(src+'/'+name)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        name = name.split('.')[0]
        cv2.imwrite(OUTPUT_DIR+'/'+name+'-clahe.jpg', bgr)

def clahe_dir(src, OUTPUT_DIR, gridsize):
    dir_list = os.listdir(src)
    print('-----CLAHE-----')
    for dir in dir_list:
        if not os.path.isdir(OUTPUT_DIR+'/'+dir):
            os.makedirs(OUTPUT_DIR+'/'+dir)
        cls_list = os.listdir(src+'/'+dir)
        for cls in cls_list:
            if not os.path.isdir(OUTPUT_DIR+'/'+dir+'/'+cls):
                os.makedirs(OUTPUT_DIR+'/'+dir+'/'+cls)
            PATH = src+'/'+dir+'/'+cls+'/'
            OUTPUT_PATH = OUTPUT_DIR+'/'+dir+'/'+cls+'/'
            filenames = os.listdir(PATH)
            for name in filenames:
                bgr = cv2.imread(PATH+name)
                lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
                lab_planes = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))
                lab_planes[0] = clahe.apply(lab_planes[0])
                lab = cv2.merge(lab_planes)
                bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                name = name.split('.')[0]
                cv2.imwrite(OUTPUT_PATH+name+'-clahe.jpg', bgr)

def clahe_green_dir(src, OUTPUT_DIR, gridsize):
    dir_list = os.listdir(src)
    print('-----CLAHE GREEN-----')
    for dir in dir_list:
        if not os.path.isdir(OUTPUT_DIR+'/'+dir):
            os.makedirs(OUTPUT_DIR+'/'+dir)
        cls_list = os.listdir(src+'/'+dir)
        for cls in cls_list:
            if not os.path.isdir(OUTPUT_DIR+'/'+dir+'/'+cls):
                os.makedirs(OUTPUT_DIR+'/'+dir+'/'+cls)
            PATH = src+'/'+dir+'/'+cls+'/'
            OUTPUT_PATH = OUTPUT_DIR+'/'+dir+'/'+cls+'/'
            filenames = os.listdir(PATH)
            for name in filenames:
                image = cv2.imread(PATH+name)
                b,green_fundus,r = cv2.split(image)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize,gridsize))
                contrast_enhanced_green_fundus = clahe.apply(green_fundus)
                bgr = cv2.cvtColor(contrast_enhanced_green_fundus, cv2.COLOR_GRAY2BGR)
                name = name.split('.')[0]
                cv2.imwrite(OUTPUT_PATH+name+'-gclahe.jpg', bgr)


def clahe_red_dir(src, OUTPUT_DIR, gridsize):
    dir_list = os.listdir(src)
    print('-----CLAHE RED-----')
    for dir in dir_list:
        if not os.path.isdir(OUTPUT_DIR+'/'+dir):
            os.makedirs(OUTPUT_DIR+'/'+dir)
        cls_list = os.listdir(src+'/'+dir)
        for cls in cls_list:
            if not os.path.isdir(OUTPUT_DIR+'/'+dir+'/'+cls):
                os.makedirs(OUTPUT_DIR+'/'+dir+'/'+cls)
            PATH = src+'/'+dir+'/'+cls+'/'
            OUTPUT_PATH = OUTPUT_DIR+'/'+dir+'/'+cls+'/'
            filenames = os.listdir(PATH)
            for name in filenames:
                image = cv2.imread(PATH+name)
                b,g,red_fundus = cv2.split(image)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize,gridsize))
                contrast_enhanced_red_fundus = clahe.apply(red_fundus)
                bgr = cv2.cvtColor(contrast_enhanced_red_fundus, cv2.COLOR_GRAY2BGR)
                name = name.split('.')[0]
                cv2.imwrite(OUTPUT_PATH+name+'-rclahe.jpg', bgr)

def clahe_blue_dir(src, OUTPUT_DIR, gridsize):
    dir_list = os.listdir(src)
    print('-----CLAHE BLUE-----')
    for dir in dir_list:
        if not os.path.isdir(OUTPUT_DIR+'/'+dir):
            os.makedirs(OUTPUT_DIR+'/'+dir)
        cls_list = os.listdir(src+'/'+dir)
        for cls in cls_list:
            if not os.path.isdir(OUTPUT_DIR+'/'+dir+'/'+cls):
                os.makedirs(OUTPUT_DIR+'/'+dir+'/'+cls)
            PATH = src+'/'+dir+'/'+cls+'/'
            OUTPUT_PATH = OUTPUT_DIR+'/'+dir+'/'+cls+'/'
            filenames = os.listdir(PATH)
            for name in filenames:
                image = cv2.imread(PATH+name)
                blue_fundus,g,r = cv2.split(image)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize,gridsize))
                contrast_enhanced_blue_fundus = clahe.apply(blue_fundus)
                bgr = cv2.cvtColor(contrast_enhanced_blue_fundus, cv2.COLOR_GRAY2BGR)
                name = name.split('.')[0]
                cv2.imwrite(OUTPUT_PATH+name+'-bclahe.jpg', bgr)

def clahe_concat(src, OUTPUT_DIR, gridsize):
    dir_list = os.listdir(src)
    for dir in dir_list:
        if not os.path.isdir(OUTPUT_DIR+'/'+dir):
            os.makedirs(OUTPUT_DIR+'/'+dir)
        cls_list = os.listdir(src+'/'+dir)
        for cls in cls_list:
            if not os.path.isdir(OUTPUT_DIR+'/'+dir+'/'+cls):
                os.makedirs(OUTPUT_DIR+'/'+dir+'/'+cls)
            PATH = src+'/'+dir+'/'+cls+'/'
            OUTPUT_PATH = OUTPUT_DIR+'/'+dir+'/'+cls+'/'
            filenames = os.listdir(PATH)
            for name in filenames:
                image = cv2.imread(PATH+name)
                blue_fundus,green_fundus,red_fundus = cv2.split(image)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize,gridsize))
                contrast_enhanced_green_fundus = clahe.apply(green_fundus)
                contrast_enhanced_blue_fundus = clahe.apply(blue_fundus)
                contrast_enhanced_red_fundus = clahe.apply(red_fundus)
                bgr_g = cv2.cvtColor(contrast_enhanced_green_fundus, cv2.COLOR_GRAY2BGR)
                bgr_b = cv2.cvtColor(contrast_enhanced_blue_fundus, cv2.COLOR_GRAY2BGR)
                bgr_r = cv2.cvtColor(contrast_enhanced_red_fundus, cv2.COLOR_GRAY2BGR)
                im_concat = cv2.vconcat([bgr_r,bgr_g,bgr_b])
                name = name.split('.')[0]
                cv2.imwrite(OUTPUT_PATH+name+'-concate.jpg', im_concat)


def red_dir(src, OUTPUT_DIR, gridsize):
    dir_list = os.listdir(src)
    for dir in dir_list:
        if not os.path.isdir(OUTPUT_DIR+'/'+dir):
            os.makedirs(OUTPUT_DIR+'/'+dir)
        cls_list = os.listdir(src+'/'+dir)
        for cls in cls_list:
            if not os.path.isdir(OUTPUT_DIR+'/'+dir+'/'+cls):
                os.makedirs(OUTPUT_DIR+'/'+dir+'/'+cls)
            PATH = src+'/'+dir+'/'+cls+'/'
            OUTPUT_PATH = OUTPUT_DIR+'/'+dir+'/'+cls+'/'
            filenames = os.listdir(PATH)
            for name in filenames:
                image = cv2.imread(PATH+name)
                b,g,red_fundus = cv2.split(image)
                bgr = cv2.cvtColor(red_fundus, cv2.COLOR_GRAY2BGR)
                name = name.split('.')[0]
                cv2.imwrite(OUTPUT_PATH+name+'-r.jpg', bgr)

def green_dir(src, OUTPUT_DIR, gridsize):
    dir_list = os.listdir(src)
    for dir in dir_list:
        if not os.path.isdir(OUTPUT_DIR+'/'+dir):
            os.makedirs(OUTPUT_DIR+'/'+dir)
        cls_list = os.listdir(src+'/'+dir)
        for cls in cls_list:
            if not os.path.isdir(OUTPUT_DIR+'/'+dir+'/'+cls):
                os.makedirs(OUTPUT_DIR+'/'+dir+'/'+cls)
            PATH = src+'/'+dir+'/'+cls+'/'
            OUTPUT_PATH = OUTPUT_DIR+'/'+dir+'/'+cls+'/'
            filenames = os.listdir(PATH)
            for name in filenames:
                image = cv2.imread(PATH+name)
                b,green_fundus,r = cv2.split(image)
                bgr = cv2.cvtColor(green_fundus, cv2.COLOR_GRAY2BGR)
                name = name.split('.')[0]
                cv2.imwrite(OUTPUT_PATH+name+'-g.jpg', bgr)

def blue_dir(src, OUTPUT_DIR, gridsize):
    dir_list = os.listdir(src)
    for dir in dir_list:
        if not os.path.isdir(OUTPUT_DIR+'/'+dir):
            os.makedirs(OUTPUT_DIR+'/'+dir)
        cls_list = os.listdir(src+'/'+dir)
        for cls in cls_list:
            if not os.path.isdir(OUTPUT_DIR+'/'+dir+'/'+cls):
                os.makedirs(OUTPUT_DIR+'/'+dir+'/'+cls)
            PATH = src+'/'+dir+'/'+cls+'/'
            OUTPUT_PATH = OUTPUT_DIR+'/'+dir+'/'+cls+'/'
            filenames = os.listdir(PATH)
            for name in filenames:
                image = cv2.imread(PATH+name)
                blue_fundus,g,r = cv2.split(image)
                bgr = cv2.cvtColor(blue_fundus, cv2.COLOR_GRAY2BGR)
                name = name.split('.')[0]
                cv2.imwrite(OUTPUT_PATH+name+'-b.jpg', bgr)

def rgb_concat(src, OUTPUT_DIR, gridsize):
    dir_list = os.listdir(src)
    for dir in dir_list:
        if not os.path.isdir(OUTPUT_DIR+'/'+dir):
            os.makedirs(OUTPUT_DIR+'/'+dir)
        cls_list = os.listdir(src+'/'+dir)
        for cls in cls_list:
            if not os.path.isdir(OUTPUT_DIR+'/'+dir+'/'+cls):
                os.makedirs(OUTPUT_DIR+'/'+dir+'/'+cls)
            PATH = src+'/'+dir+'/'+cls+'/'
            OUTPUT_PATH = OUTPUT_DIR+'/'+dir+'/'+cls+'/'
            filenames = os.listdir(PATH)
            for name in filenames:
                image = cv2.imread(PATH+name)
                blue_fundus,green_fundus,red_fundus = cv2.split(image)
                bgr_g = cv2.cvtColor(green_fundus, cv2.COLOR_GRAY2BGR)
                bgr_b = cv2.cvtColor(blue_fundus, cv2.COLOR_GRAY2BGR)
                bgr_r = cv2.cvtColor(red_fundus, cv2.COLOR_GRAY2BGR)
                im_concat = cv2.vconcat([bgr_r,bgr_g,bgr_b])
                name = name.split('.')[0]
                cv2.imwrite(OUTPUT_PATH+name+'-concatergb.jpg', im_concat)

if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    src = '/home/psj/Desktop/3ycut15_data_fazekas/trainvaltest/val'
    OUTPUT_DIR = '/home/psj/Desktop/3ycut15_data_fazekas/testset_clahe'
    dir_list = os.listdir(src)
    for dir in dir_list:
         clahe_allfiles_dir(src+'/'+dir, OUTPUT_DIR+'/'+dir, 8)
    '''
    for dir in dir_list:
        if not os.path.isdir(OUTPUT_DIR+'/'+dir):
            os.makedirs(OUTPUT_DIR+'/'+dir)
        cls_list = os.listdir(src+'/'+dir)
        for cls in cls_list:
            if not os.path.isdir(OUTPUT_DIR+'/'+dir+'/'+cls):
                os.makedirs(OUTPUT_DIR+'/'+dir+'/'+cls)
            PATH = src+'/'+dir+'/'+cls+'/'
            OUTPUT_PATH = OUTPUT_DIR+'/'+dir+'/'+cls+'/'
            filenames = os.listdir(PATH)
            for name in filenames:
                img=clahe(PATH+name,7)
                #cv2.imwrite(OUTPUT_PATH+name+'-CLAHE.jpg', img)
                cv2.imwrite(OUTPUT_PATH+name, img)
    '''
