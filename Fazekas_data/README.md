## preprocess.py
***
train / validation split
***
다음 중 필요한 부분을 선택하여 사용
1. Resize
<pre>
<code>
    #resize_allfiles(in_path, out_path)
    #resize
    resize_dir(in_path, out_path)
</code>
</pre>
2. Train Validation Split
<pre>
<code>
    #train_validation_split
    in_path = out_path
    out_path = out_path + '_split'
    rand_seed = 888
    #trainval_split(in_path, out_path, rand_seed)
    groupsplit(in_path, out_path, rand_seed, 8/9)
</code>
</pre>
trainval_split : 이미지 수 가준 split
groupsplit : 환자 수 기준 split
