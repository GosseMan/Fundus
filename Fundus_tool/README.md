optic_extract.py
==============
참고 : <https://doi.org/10.1016/j.cmpb.2015.10.010>
***
<img src="./OD_example/1.jpg" width="30%" height="10%"> <img src="./OD_example/2.jpg" width="30%" height="10%"> <img src="./OD_example/3.jpg" width="30%" height="10%">
<img src="./OD_example/4.jpg" width="30%" height="10%"> <img src="./OD_example/5.jpg" width="30%" height="10%"> <img src="./OD_example/6.jpg" width="30%" height="10%">
##### 1.Original Image
##### 2.Find optic disk position
Kaiser Window 적용시킨 후 사진 내에서 가장 밝은 지점
##### 3.Crop circle including optic disk
2에서 찾은 점을 중심으로 optic disk 포함하여 crop
##### 4.Draw contour
crop된 부분 binarization(특정 Threshold 지정가능) -> contour 
##### 5.Find fit-ellipse
contour의 fit-ellipse를 찾고 contour의 면적과 fit-ellipse의 면적비율(circularity)에 따라 optic disk로 채택
##### 6.Optic disk result
