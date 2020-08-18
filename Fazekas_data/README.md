# preprocess.py
***
## train / validation split
***
다음 중 필요한 부분을 선택하여 사용 (각 단계별로 out_path에 결과물 저장)
### 1. Resize
<pre>
<code>
    #resize_allfiles(in_path, out_path)
    #resize
    resize_dir(in_path, out_path)
</code>
</pre>

### 2. Train Validation Split
<pre>
<code>
    #train_validation_split
    in_path = out_path
    out_path = out_path + '_split'
    rand_seed = 8
    #trainval_split(in_path, out_path, rand_seed)
    groupsplit(in_path, out_path, rand_seed, 0.9)
</code>
</pre>
trainval_split : 이미지 수 가준 split\
groupsplit : 환자 수 기준 split

### 3. Undersampling
<pre>
<code>
    #undersampling
    in_path = out_path
    out_path = out_path + '_under'
    under_class_list = ['0ZERO', '1ONE']
    num_sample_list = [100, 50]
    undersampling(in_path,out_path,under_class_list,num_sample_list,rand_seed)
</code>
</pre>
under_class_list : undersample할 class명\
num_sample_list : undersample 개수\
-> 위의 경우 '0ZERO' class 100개, '1ONE' class 50개로 undersample

### 4. Zoom
<pre>
<code>
    #zoom
    in_path = out_path
    out_path = out_path +'_zoom'
    shutil.copytree(in_path, out_path)
    class_list_zoom = ['1ONE', '2TWO']
    zoom_list = [1.1, 1.2]
    filter = cv2.imread('/home/psj/Desktop/psj_util/560X480/threshold40.jpg')
    filter = filter/255
    for zoom in zoom_list:
        for cls in class_list_zoom:
            zoom_use_filter(in_path+'/train/'+cls,out_path+'/train/'+cls, zoom, filter)
</code>
</pre>
class_list_zoom : zoom할 class명\
zoom_list : zoom 비율\
filter : 기존에 저장된 사진 filter\
-> 위의 경우 '1ONE' class, '2TWO' class 모두 1.1, 1.2배 zoom

### 5. Augmentation
<pre>
<code>
    #augmentation
    in_path = out_path
    out_path = out_path + '_aug'
    class_list = ['0ZERO', '1ONE', '2TWO']
    shutil.copytree(in_path, out_path)
    aug = ApplyAug()
    for cls in class_list:
        aug.applyAug(in_path+'/train/'+cls+'/', out_path+'/train/'+cls+'/')
        #aug.applyAugLRUD(out_path+'/train/'+cls+'/', out_path+'/train/'+cls+'/')
</code>
</pre>
class_list : augmentation할 class명\
aug.py에서 지정한 augmentation 수행(현재 vertical flip)

### 6. CLAHE
<pre>
<code>
    #CLAHE
    in_path = out_path
    out_path = out_path+'_clahe'
    clahe.clahe_dir(in_path,out_path,8)
    #clahe.clahe_green_dir(in_path, out_path, 8)
    #clahe.clahe_red_dir(in_path, out_path, 8)
    #clahe.clahe_blue_dir(in_path, out_path, 8)
    #clahe.clahe_concat(in_path, out_path, 8)
    #clahe.rgb_concat(in_path,out_path,8)
</code>
</pre>

