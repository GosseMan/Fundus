import cpbd
import cv2
import numpy as np
import os
import shutil
from glob import glob
import matplotlib.pyplot as plt
from scipy import ndimage
import skimage.io
def vessel_density(in_path):
    file_list = os.listdir(in_path)
    density_list = []
    for file in file_list:
        img = cv2.imread(in_path+'/'+file, cv2.IMREAD_GRAYSCALE)
        density_list.append(round(np.average(img),2))
    return np.array(density_list), np.array(file_list)

def vessel_density_dir(in_path, vessel_path):
    file_list = os.listdir(in_path)
    density_list = []
    for file in file_list:
        img = cv2.imread(vessel_path+'/'+file, cv2.IMREAD_GRAYSCALE)
        density_list.append(round(np.average(img),2))
    return np.array(density_list), np.array(file_list)


def cpbd_check(in_path):
    file_list = os.listdir(in_path)
    cpbd_list = []
    for file in file_list:
        print(in_path+'/'+file)
        img = cv2.imread(in_path+'/'+file, cv2.IMREAD_GRAYSCALE)

        cpbd_list.append(round(cpbd.compute(img),2))
        #print(cpbd.compute(img))
    return np.array(cpbd_list), np.array(file_list)

def main():
    vessel_path = '/home/psj/Desktop/data_fazekas/3years_allfiles_vessel'
    origin_path = '/home/psj/Desktop/data_fazekas/3years_allfiles'
    #green_path = '/home/psj/Desktop/mFS/mFS_allfiles_green'
    vessel_output_path = '/home/psj/Desktop/data_fazekas/3years_allfiles_vessel_quality'
    #cpbd_output_path = '/home/psj/Desktop/mFS/mFS_cpbd_quality'
    if not os.path.isdir(vessel_output_path):
        os.makedirs(vessel_output_path)

    '''
    file_list = os.listdir(origin_path)
    for file in file_list:
        print(origin_path+'/'+file)
        image = cv2.imread(origin_path+'/'+file)
        b,green_fundus,r = cv2.split(image)
        bgr = cv2.cvtColor(green_fundus, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(green_path+'/'+file, bgr)
    '''

    #vessel density check

    density_list, file_list = vessel_density(vessel_path)
    #print(density_list)
    ascending_indices = density_list.argsort()
    i = 0
    for idx in ascending_indices:
        src = origin_path+'/'+str(file_list[idx])
        output = vessel_output_path +'/'+ str(i)+'-'+str(density_list[idx])+'-'+str(file_list[idx])

        shutil.copy(src,output)
        i=i+1
    plt.hist(density_list, 15)
    #plt.savefig('./histogram_vesseldensity2')
    plt.close()
    '''
    vessel_dir_path = '/home/psj/Desktop/mFS/mFS_2years_ascend_vesseldensity'
    dir_list = os.listdir(origin_path)
    for dir in dir_list:
        if not os.path.isdir(vessel_dir_path+'/'+dir):
            os.makedirs(vessel_dir_path+'/'+dir)
        src_path = origin_path+'/'+dir
        density_list, file_list = vessel_density_dir(src_path, vessel_path)
        print(len(density_list))
        print(len(file_list))
        ascending_indices = density_list.argsort()
        i = 0
        for idx in ascending_indices:
            src = origin_path+'/'+dir+'/'+str(file_list[idx])
            output = vessel_dir_path +'/'+dir+'/'+ str(i)+'-'+str(density_list[idx])+'-'+str(file_list[idx])

            shutil.copy(src,output)
            i=i+1
        plt.hist(density_list, 15)
        plt.savefig('./'+dir+'.png')
        plt.close()
    '''
    '''
    #cpbd check
    if not os.path.isdir(cpbd_output_path):
        os.makedirs(cpbd_output_path)
    cpbd_list, file_list = cpbd_check(green_path)
    print(cpbd_list)
    ascending_indices = cpbd_list.argsort()
    i = 0
    for idx in ascending_indices:
        src = origin_path+'/'+str(file_list[idx])
        output = cpbd_output_path +'/'+ str(i)+'-'+str(cpbd_list[idx])+'-'+str(file_list[idx])
        shutil.copy(src,output)
        i=i+1
    plt.hist(density_list, 15)
    plt.savefig('./histogram_cpbd')
    '''

if __name__ == "__main__":
    main()
