from undersampling import undersampling
from resize import resize_dir, resize_allfiles
from trainvalsplit import trainval_split
from aug import ApplyAug
from zoomin import zoom_use_filter
import shutil
from groupsplit import groupsplit
import cv2
import clahe
import os

if __name__ == "__main__":
    in_path = 'input path'
    out_path = 'ouput path'

    #resize_allfiles(in_path, out_path)
    #resize
    resize_dir(in_path, out_path)

    #train_validation_split
    in_path = out_path
    out_path = out_path + '_split'
    rand_seed = 8
    #trainval_split(in_path, out_path, rand_seed)
    groupsplit(in_path, out_path, rand_seed, 0.9)

    #undersampling
    in_path = out_path
    out_path = out_path + '_under'
    under_class_list = ['0ZERO', '1ONE']
    num_sample_list = [100, 50]
    undersampling(in_path,out_path,under_class_list,num_sample_list,rand_seed)


    #zoom
    in_path = out_path
    out_path = out_path +'_zoom'
    shutil.copytree(in_path, out_path)
    class_list_zoom = ['2TWO']
    zoom_list = [1.1, 1.2]
    filter = cv2.imread('/home/psj/Desktop/psj_util/560X480/threshold40.jpg')
    filter = filter/255
    for zoom in zoom_list:
        for cls in class_list_zoom:
            zoom_use_filter(in_path+'/train/'+cls,out_path+'/train/'+cls, zoom, filter)


    #augmentation
    in_path = out_path
    out_path = out_path + '_aug'
    class_list = ['0ZERO', '1ONE', '2TWO']
    shutil.copytree(in_path, out_path)
    aug = ApplyAug()
    for cls in class_list:
        aug.applyAug(in_path+'/train/'+cls+'/', out_path+'/train/'+cls+'/')
        #aug.applyAugLRUD(out_path+'/train/'+cls+'/', out_path+'/train/'+cls+'/')


    #CLAHE
    in_path = out_path
    out_path = out_path+'_clahe'
    clahe.clahe_dir(in_path,out_path,8)
    #clahe.clahe_green_dir(in_path, out_path, 8)
    #clahe.clahe_red_dir(in_path, out_path, 8)
    #clahe.clahe_blue_dir(in_path, out_path, 8)
    #clahe.clahe_concat(in_path, out_path, 8)
    #clahe.rgb_concat(in_path,out_path,8)
