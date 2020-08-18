import os
import numpy as np
import shutil
import random

def trainval_split(path, out_path, rand_seed):
    # # Creating Train / Val / Test folders (One time use)
    val_ratio = 0.1
    #test_ratio = 0.05
    os.makedirs(out_path+'/train')
    os.makedirs(out_path+'/val')
    classes_dir = os.listdir(path)
    #os.makedirs(root_dir+'/test')
    for cls in classes_dir:
        os.makedirs(out_path +'/train/' + cls)
        os.makedirs(out_path +'/val/' + cls)
        #os.makedirs(root_dir +'/test/' + cls)
        # Creating partitions of the data after shuffeling
        src = path + '/'+cls # Folder to copy images from

        allFileNames = os.listdir(src)
        np.random.seed(rand_seed)
        np.random.shuffle(allFileNames)
        train_FileNames, val_FileNames = np.split(np.array(allFileNames),[int(len(allFileNames)* (1 - val_ratio))])


        train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
        val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
        #test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

        print('Total images: ', len(allFileNames))
        print('Training: ', len(train_FileNames))
        print('Validation: ', len(val_FileNames))
        #print('Testing: ', len(test_FileNames))

        # Copy-pasting images
        for name in train_FileNames:
            shutil.copy(name, out_path +'/train/' +cls)

        for name in val_FileNames:
            shutil.copy(name, out_path +'/val/' + cls)

            #for name in test_FileNames:
            #shutil.copy(name, root_dir +'/test/' + cls)

def trainval_split_regression(path, out_path, rand_seed):
    # # Creating Train / Val / Test folders (One time use)
    val_ratio = 0.1
    #test_ratio = 0.05

    os.makedirs(out_path+'/train')
    os.makedirs(out_path+'/val')
    #os.makedirs(root_dir+'/test')
    #os.makedirs(root_dir +'/test/' + cls)
    # Creating partitions of the data after shuffeling
    src = path # Folder to copy images from
    allFileNames = os.listdir(src)
    np.random.seed(rand_seed)
    np.random.shuffle(allFileNames)
    train_FileNames, val_FileNames = np.split(np.array(allFileNames),[int(len(allFileNames)* (1 - val_ratio))])

    train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
    val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
    #test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]
    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    #print('Testing: ', len(test_FileNames))

    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name, out_path +'/train/')
    for name in val_FileNames:
        shutil.copy(name, out_path +'/val/')

        #for name in test_FileNames:
        #shutil.copy(name, root_dir +'/test/' + cls)
if __name__ == "__main__":
    in_path = '../data/bmi_q3_clahe_v5'
    out_path = '../data/bmi_q3_clahe_split_v5'

    rand_seed = 7
    trainval_split(in_path, out_path, rand_seed)
