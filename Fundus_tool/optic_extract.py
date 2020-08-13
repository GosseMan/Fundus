import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import math
from mpl_toolkits.mplot3d.axes3d import Axes3D


def kaiser_filtering(k_filter, green_im):
    f = cv2.dft(green_im.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    f_shifted = np.fft.fftshift(f)
    f_complex = f_shifted[:,:,0]*1j + f_shifted[:,:,1]
    f_filtered = k_filter * f_complex
    f_filtered_shifted = np.fft.fftshift(f_filtered)
    inv_img = np.fft.ifft2(f_filtered_shifted) # inverse F.T.
    filtered_img = np.abs(inv_img)
    return filtered_img

def center_detection(img_path):
    img_name = img_path.split('/')[-1]
    rgb_img = cv2.imread(img_path)
    cv2.imwrite('../../small/1.jpg',rgb_img)
    ser = img_name.split('.')[0]
    od_box = int(rgb_img.shape[1]/20)
    blue, green, red = cv2.split(rgb_img)
    kaiser1 = np.kaiser(od_box,14)[:,None] # 1D hamming
    kaiser2 = np.kaiser(od_box,14)[:,None]
    kaiser2d = np.sqrt(np.dot(kaiser1, kaiser2.T)) # expand to 2D hamming
    k_filter = kaiser1
    best = 0
    for i in range(rgb_img.shape[1]-1):
        k_filter=np.hstack((k_filter,kaiser1))
    for j in range(int(rgb_img.shape[0]/od_box)):
        if j*od_box+od_box>rgb_img.shape[0]+1:
            break
        else:
            green_im=green[j*od_box:j*od_box+od_box,:]
        filtered_img = kaiser_filtering(k_filter,green_im)
        if np.max(filtered_img)>best:
            best = np.max(filtered_img)
            y_index = j*od_box
    green_im=green[y_index:y_index+od_box,:]
    horizon_filtered_img = kaiser_filtering(k_filter,green_im)
    best = 0
    for i in range(int(rgb_img.shape[1]/od_box)):
        if j*od_box+od_box>rgb_img.shape[0]+1:
            break
        else:
            green_im=green[y_index:y_index+od_box,i*od_box:i*od_box+od_box]
        filtered_img = kaiser_filtering(kaiser2d, green_im)
        if np.max(filtered_img)>best:
                best = np.max(filtered_img)
                x_index = i*od_box
    green_im=green[y_index:y_index+od_box:,x_index:x_index+od_box]
    beste_filtered_img = kaiser_filtering(kaiser2d, green_im)

    center_x = int(x_index+od_box/2)
    center_y = int(y_index+od_box/2)
    center = cv2.line(rgb_img, (center_x,center_y),(center_x,center_y), (255,0,0), 20)
    cv2.imwrite('../../small/2.jpg',center)
    return center_x, center_y

def od_cropping(img_path, out_path, center_x, center_y, r):
    rgb_img = cv2.imread(img_path)
    gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    blue, green, red = cv2.split(rgb_img)
    img_name = img_path.split('/')[-1]
    cv2.circle(rgb_img, (center_x,center_y), r, (0,0,0), 2)

    cv2.imwrite('../../small/3.jpg',rgb_img)
    mask = np.zeros((rgb_img.shape[0], rgb_img.shape[1]), dtype=np.uint8)
    cv2.circle(mask, (center_x,center_y), r, (1,1,1), -1, 15, 0)
    cropped = red*mask
    cond=np.where(cropped>0)
    dif = np.max(cropped[cond])-np.min(cropped[cond])
    cropped_normal = (red-np.min(cropped[cond]))*(255/dif)*mask
    threshold = 140
    while True:
        if threshold > 255:
            print('Cannot find OD :', img_path)
            return (0,0)
        binary_img = (cropped_normal>=threshold).astype('uint8')*255
        contours, hierachy = cv2.findContours(binary_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)
        if len(contours) == 0:
            threshold = threshold+10
            continue
        best_cnt = contours[0]
        best_area = 0
        for cnt, hier in zip(contours, hierachy[0]):
            temp_area = cv2.contourArea(cnt)
            if hier[3]==-1 and temp_area>best_area and cv2.pointPolygonTest(cnt,(center_x,center_y),True)>=0:
                best_cnt = cnt
                best_area = temp_area
        if best_cnt.shape[0] < 10: #at least contour shout have 10 points
            threshold = threshold+10
            continue
        (x,y), (MA, ma), angle = cv2.fitEllipse(best_cnt)
        fit_ellipse_area = math.pi * MA * ma/4
        circularity = best_area/fit_ellipse_area
        cv2.drawContours(rgb_img, [best_cnt], 0, (0, 255, 0), 1)
        cv2.imwrite('../../small/4.jpg',rgb_img)
        if best_area > mask.sum()/4 or ma/MA>1.5:
            threshold = threshold+10
            continue
        else:
            circularity = best_area/fit_ellipse_area
        #print(circularity)
        if 0.95 < circularity < 1.05:
            break
        threshold = threshold+10
    el_img =  cv2.ellipse(rgb_img, ((x,y), (MA, ma), angle), (255, 255, 255), -1) #red
    cv2.imwrite('../../small/5.jpg',el_img)
    od_mask = (el_img==255)
    od_mask = od_mask.astype(np.uint8)
    od = od_mask[:,:,0] * gray_img
    cv2.imwrite(out_path+img_name,od)


def extract_od(img_path, out_path):
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    rgb_img = cv2.imread(img_path)
    #algorithm1
    center_x, center_y = center_detection(img_path)
    #algorithm2
    r= int(rgb_img.shape[0]/5)
    od_cropping(img_path, out_path, center_x, center_y, r)

def main():
    #Input Image Path example
    file = '../../small/vk034394.jpg'
    #Output Image Path example
    output_path = "../../small/"
    extract_od(file,output_path)
    return

if __name__ == "__main__":
	main()
