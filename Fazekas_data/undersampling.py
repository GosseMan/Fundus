import os
import numpy as np
import shutil
import random

def undersampling(in_path, out_path, under_class_list, num_sample_list, rand_seed):
    # # Creating Train / Val / Test folders (One time use)
    #test_ratio = 0.05
    #os.makedirs(root_dir+'/test')
    shutil.copytree(in_path+'/val', out_path+'/val')
    dir_list = os.listdir(in_path+'/train')
    for dir in dir_list:
        if not dir in under_class_list:
            shutil.copytree(in_path+'/train/'+dir,out_path+'/train/'+dir)
        else:
            os.makedirs(out_path+'/train/'+dir)
            src = in_path+'/train/'+dir
            allFileNames = os.listdir(src)
            np.random.seed(rand_seed)
            np.random.shuffle(allFileNames)
            num_sample = num_sample_list[under_class_list.index(dir)]
            train_FileNames, trash = np.split(np.array(allFileNames),[num_sample])
            train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
            #test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]
            print('Total images: ', len(allFileNames))
            print('Training: ', len(train_FileNames))
            print('Trash: ', len(trash))
            #print('Testing: ', len(test_FileNames))
            # Copy-pasting images
            for name in train_FileNames:
                shutil.copy(name, out_path +'/train/' + dir)

        #for name in test_FileNames:
            #shutil.copy(name, root_dir +'/test/' + cls)
if __name__ == "__main__":
    in_path = '../mFS/mFS_2years_seed777_split'
    out_path = '../mFS/mFS_2years_seed777_split_under'
    under_class_list = ['0ZERO', '1ONE']
    rand_seed = 7
    num_sample_list = [450, 450]
    undersampling(in_path, out_path, under_class_list, num_sample_list, rand_seed)
