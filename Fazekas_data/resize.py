import os
from glob import glob
from PIL import Image
def resize_dir(path,outpath):
    dir_list = os.listdir(path)
    for dir in dir_list:
        dir_path = path+'/'+dir
        file_list = os.listdir(dir_path)
        outpath_dir = outpath +'/'+dir
        try:
            if not os.path.exists(outpath_dir):
                os.makedirs(outpath_dir)
        except OSError:
            print('Error: creating '+ outpath_dir)
        for file in file_list:
            image = Image.open(dir_path+'/'+file)
            resize_image = image.resize((1920,1296))
            area = (207,0,1719,1296)
            resize_image = resize_image.crop(area)
            resize_image = resize_image.resize((560,480))
            resize_image.save(outpath_dir+'/'+file,"JPEG",quality=100)
    return outpath

def resize_allfiles(path,outpath):
    file_list = os.listdir(path)
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
    for file in file_list:
        image = Image.open(path+'/'+file)
        resize_image = image.resize((1920,1296))
        area = (207,0,1719,1296)
        resize_image = resize_image.crop(area)
        resize_image = resize_image.resize((560,480))
        resize_image.save(outpath+'/'+file,"JPEG",quality=100)
    return outpath

def resize_file():
    path = '/home/psj/Desktop/fundus_data/all_files'
    file_list = os.listdir(path)
    newpath = '/home/psj/Desktop/FUNDUS_DATA_480/all_files'
    try:
        if not os.path.exists(newpath):
            os.makedirs(newpath)
    except OSError:
        print('Error: creating '+ newpath)
    for file in file_list:
        image = Image.open(path+'/'+file)
        resize_image = image.resize((480,480))
        resize_image.save(newpath+'/'+file,"JPEG",quality=100)

if __name__ == "__main__":
    path = '/home/psj/Desktop/ex'
    out_path = '/home/psj/Desktop/ex_resized'

    resize_dir(path,out_path)
