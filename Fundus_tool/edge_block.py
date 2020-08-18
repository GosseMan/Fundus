# coding: utf-8

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals
)

from math import atan2, pi
from sys import argv

import numpy as np
from scipy import ndimage
import skimage
from skimage.feature import canny
import cv2
import os
from cpbd.octave import sobel
import matplotlib.pyplot as plt
import shutil
# threshold to characterize blocks as edge/non-edge blocks
THRESHOLD = 0.02
# block size
BLOCK_HEIGHT, BLOCK_WIDTH = (8, 8)

def edgeblock_num(edges):
    img_height, img_width = edges.shape
    num_blocks_vertically = int(img_height / BLOCK_HEIGHT)
    num_blocks_horizontally = int(img_width / BLOCK_WIDTH)
    c=0
    for i in range(num_blocks_vertically):
        for j in range(num_blocks_horizontally):
            rows = slice(BLOCK_HEIGHT * i, BLOCK_HEIGHT * (i + 1))
            cols = slice(BLOCK_WIDTH * j, BLOCK_WIDTH * (j + 1))
            if is_edge_block(edges[rows, cols], THRESHOLD):
                c=c+1
    return c

def is_edge_block(block, threshold):
    """Decide whether the given block is an edge block."""
    return np.count_nonzero(block) > (block.size * threshold)


if __name__ == '__main__':
        path ='/home/psj/Desktop/small'
        sort_path='/home/psj/Desktop/small_edgeblock'
        file_list = os.listdir(path)
        edge_list = []
        if not os.path.isdir(sort_path):
            os.makedirs(sort_path)
        for file in file_list:
            rgb_image = cv2.imread(path+'/'+file)
            b,green,r = cv2.split(rgb_image)
            canny_edges = canny(green, low_threshold = 20, high_threshold = 25)
            cv2.imwrite(sort_path+'/canny-'+file,canny_edges)
            edge_block = edgeblock_num(canny_edges)
            edge_list.append(edge_block)

        ascending_indices = np.array(edge_list).argsort()
        i = 0
        for idx in ascending_indices:
            src = path+'/'+str(file_list[idx])
            output = sort_path +'/'+ str(i)+'-'+str(edge_list[idx])+'-'+str(file_list[idx])
            shutil.copy(src,output)
            i=i+1
