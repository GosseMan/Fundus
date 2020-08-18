import numpy as np
import os
import shutil
from datetime import datetime
from sklearn.model_selection import GroupShuffleSplit

def groupsplit(data_path, out_path, rand_seed, train_percent):

    cls_list = os.listdir(data_path)
    for cls in cls_list:
        file_list = os.listdir(data_path+'/'+cls)
        y = [0]*len(file_list)
        id_list = []
        for file in file_list:
            #print(file)
            id_list.append(file.split('-')[1])
        gss = GroupShuffleSplit(n_splits = 1, train_size = train_percent, random_state=rand_seed)
        #print("TRAIN: ",train_file, "TEST: ",test_file)

        for train_file, test_file in gss.split(file_list,y,id_list):
            print(cls + '-> Train : ' +str(len(train_file))+ ' , Test : '+str(len(test_file)))
            print('-----GroupShuffleSplit-----')
            train_list = train_file
            test_list = test_file
        train_file_list=[]
        test_file_list=[]
        if not os.path.isdir(out_path+'/train/'+cls):
            os.makedirs(out_path+'/train/'+cls)
        if not os.path.isdir(out_path+'/val/'+cls):
            os.makedirs(out_path+'/val/'+cls)

        for train_idx in train_list:
            train_file_list.append(file_list[train_idx])
            shutil.copy(data_path+'/'+cls+'/'+file_list[train_idx],out_path+'/train/'+cls)
        for test_idx in test_list:
            test_file_list.append(file_list[test_idx])
            shutil.copy(data_path+'/'+cls+'/'+file_list[test_idx],out_path+'/val/'+cls)

def groupsplit_samenum(data_path, out_path, rand_seed, train_percent):
    cls_list = os.listdir(data_path)


    cls = '2TWO'
    file_list = os.listdir(data_path+'/'+cls)
    y = [0]*len(file_list)
    id_list = []
    for file in file_list:
        #print(file)
        id_list.append(file.split('-')[1])
    gss = GroupShuffleSplit(n_splits = 1, train_size = train_percent, random_state=rand_seed)
    #print("TRAIN: ",train_file, "TEST: ",test_file)
    for train_file, test_file in gss.split(file_list,y,id_list):
        print(cls + '-> Train : ' +str(len(train_file))+ ' , Test : '+str(len(test_file)))
        print('-----GroupShuffleSplit-----')
        train_list = train_file
        test_list = test_file
    train_file_list=[]
    test_file_list=[]
    if not os.path.isdir(out_path+'/train/'+cls):
        os.makedirs(out_path+'/train/'+cls)
    if not os.path.isdir(out_path+'/val/'+cls):
        os.makedirs(out_path+'/val/'+cls)

    for train_idx in train_list:
        train_file_list.append(file_list[train_idx])
        shutil.copy(data_path+'/'+cls+'/'+file_list[train_idx],out_path+'/train/'+cls)
    for test_idx in test_list:
        test_file_list.append(file_list[test_idx])
        shutil.copy(data_path+'/'+cls+'/'+file_list[test_idx],out_path+'/val/'+cls)
    '''
    id_crit_list = []
    file_crit_list = os.listdir(out_path+'/train/'+cls)
    for file in file_crit_list:
        #print(file)
        id_crit_list.append(file.split('-')[1])
    crit_num = len(list(set(id_crit_list)))
    '''
    crit_num = len(list(set(id_list)))
    cls_list = ['0ZERO','1ONE']
    for cls in cls_list:
        file_list = os.listdir(data_path+'/'+cls)
        y = [0]*len(file_list)
        id_list = []
        for file in file_list:
            #print(file)
            id_list.append(file.split('-')[1])
        id_num = len(list(set(id_list)))
        print(crit_num)
        print(id_num)
        print(1-(1-train_percent)*(crit_num/id_num))
        gss = GroupShuffleSplit(n_splits = 1, train_size = 1-(1-train_percent)*(crit_num/id_num), random_state=rand_seed)
        #print("TRAIN: ",train_file, "TEST: ",test_file)

        for train_file, test_file in gss.split(file_list,y,id_list):
            print(cls + '-> Train : ' +str(len(train_file))+ ' , Test : '+str(len(test_file)))
            print('-----GroupShuffleSplit-----')
            train_list = train_file
            test_list = test_file
        train_file_list=[]
        test_file_list=[]
        if not os.path.isdir(out_path+'/train/'+cls):
            os.makedirs(out_path+'/train/'+cls)
        if not os.path.isdir(out_path+'/val/'+cls):
            os.makedirs(out_path+'/val/'+cls)

        for train_idx in train_list:
            train_file_list.append(file_list[train_idx])
            shutil.copy(data_path+'/'+cls+'/'+file_list[train_idx],out_path+'/train/'+cls)
        for test_idx in test_list:
            test_file_list.append(file_list[test_idx])
            shutil.copy(data_path+'/'+cls+'/'+file_list[test_idx],out_path+'/val/'+cls)

if __name__ == "__main__":
    data_path = '../data/mFS_2years'
    out_path = '../data/mFS_2years_split'
    rand_seed = 7
    groupsplit(data_path, out_path, rand_seed, 0.85)
