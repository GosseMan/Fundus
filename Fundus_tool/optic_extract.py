import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import math
from mpl_toolkits.mplot3d.axes3d import Axes3D

'''
def optic_detect(img_path,out_path):
    img_name = img_path.split('/')[-1]
    img = cv2.imread(img_path)
    gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    origin = img
    print(img_path)
    ser = img_name.split('.')[0]
    cv2.imwrite(out_path+ser+'-원본-'+img_name,img)
    #od box는 height 와 width의 공약수로
    od_box = 80
    blue, green, red = cv2.split(img)
    cv2.imwrite(out_path+ser+'-green-'+img_name,green)
    cv2.imwrite(out_path+ser+'-red-'+img_name,red)
    kaiser1 = np.kaiser(od_box,14)[:,None] # 1D hamming
    kaiser2 = np.kaiser(od_box,14)[:,None]
    kaiser2d = np.sqrt(np.dot(kaiser1, kaiser2.T)) # expand to 2D hamming
    k_filter = kaiser1
    best = 0
    x_index = 0

    for i in range(img.shape[1]-1):
        k_filter=np.hstack((k_filter,kaiser1))

    #cv2.imwrite('./od/aaa.jpg',k_filter*255)

    for j in range(int(img.shape[0]/od_box)):
        if j*od_box+od_box>img.shape[0]+1:
            break
        else:
            green_im=green[j*od_box:j*od_box+od_box,:]
        f = cv2.dft(green_im.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
        f_shifted = np.fft.fftshift(f)
        #four=((f_shifted[:,:,0]**2 + f_shifted[:,:,1]**2)**(1/2))/255
        #cv2.imwrite('./fft.jpg',four)
        f_complex = f_shifted[:,:,0]*1j + f_shifted[:,:,1]
        f_filtered = k_filter * f_complex
        f_filtered_shifted = np.fft.fftshift(f_filtered)
        inv_img = np.fft.ifft2(f_filtered_shifted) # inverse F.T.
        filtered_img = np.abs(inv_img)




        #print(str(j)+'->'+str(filtered_img.sum()))
        if np.max(filtered_img)>best:
                best = np.max(filtered_img)
                x_index = j*od_box

        #cv2.imwrite('./od/origin'+str(j)+'.jpg',green_im)
        #cv2.imwrite('./od/filtered'+str(j)+'.jpg',filtered_img)

    green_im=green[x_index:x_index+od_box,:]
    f = cv2.dft(green_im.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    f_shifted = np.fft.fftshift(f)
    #four=((f_shifted[:,:,0]**2 + f_shifted[:,:,1]**2)**(1/2))/255
    #cv2.imwrite('./fft.jpg',four)
    f_complex = f_shifted[:,:,0]*1j + f_shifted[:,:,1]
    f_filtered = k_filter * f_complex
    f_filtered_shifted = np.fft.fftshift(f_filtered)
    inv_img = np.fft.ifft2(f_filtered_shifted) # inverse F.T.
    filtered_img = np.abs(inv_img)
    print('best : '+str(x_index)+'~'+str(x_index+od_box-1))
    print('X-coordinate of Optic Disk Center : ',str(x_index+(od_box-1)/2))
    #cv2.imwrite(out_path+ser+'-best-'+img_name,filtered_img)
    #x_best = filtered_img
    #b,green_x,r = cv2.split(x_best)
    best = 0
    y_index = 0
    for i in range(int(img.shape[1]/od_box)):
        if j*od_box+od_box>img.shape[0]+1:
            break
        else:
            green_im=green[x_index:x_index+od_box,i*od_box:i*od_box+od_box]
        f = cv2.dft(green_im.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
        f_shifted = np.fft.fftshift(f)
        #four=((f_shifted[:,:,0]**2 + f_shifted[:,:,1]**2)**(1/2))/255
        #cv2.imwrite('./fft.jpg',four)
        f_complex = f_shifted[:,:,0]*1j + f_shifted[:,:,1]
        f_filtered = kaiser2d * f_complex
        f_filtered_shifted = np.fft.fftshift(f_filtered)
        inv_img = np.fft.ifft2(f_filtered_shifted) # inverse F.T.
        filtered_img = np.abs(inv_img)

        #print(str(j)+'->'+str(filtered_img.sum()))
        if np.max(filtered_img)>best:
                best = np.max(filtered_img)
                y_index = i*od_box
        #cv2.imwrite('./od/origin_y'+str(i)+'.jpg',green_im)
        #cv2.imwrite('./od/filtered_y'+str(i)+'.jpg',filtered_img)
    green_im=green[x_index:x_index+od_box:,y_index:y_index+od_box]
    f = cv2.dft(green_im.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    f_shifted = np.fft.fftshift(f)
    #four=((f_shifted[:,:,0]**2 + f_shifted[:,:,1]**2)**(1/2))/255
    #cv2.imwrite('./fft.jpg',four)
    f_complex = f_shifted[:,:,0]*1j + f_shifted[:,:,1]
    f_filtered = kaiser2d * f_complex
    f_filtered_shifted = np.fft.fftshift(f_filtered)
    inv_img = np.fft.ifft2(f_filtered_shifted) # inverse F.T.
    filtered_img = np.abs(inv_img)


    print('best : '+str(y_index)+'~'+str(y_index+od_box-1))
    print('Y-coordinate of Optic Disk Center : ',str(y_index+(od_box-1)/2))
    #cv2.imwrite(out_path+ser+'-bestOD-'+img_name,filtered_img)
    center_x = int(x_index+od_box/2)
    center_y = int(y_index+od_box/2)
    center = cv2.line(img, (center_y,center_x),(center_y,center_x), (255,0,0), 5)
    #cv2.imwrite('./od/center'+img_name,center)

    r=250
    cv2.circle(img,(center_y,center_x), r, (0,0,0),2)
    result = img[center_y-r:center_y+r,center_x-r:center_x+r]
    #cv2.imwrite(out_path+ser+'-center-'+img_name,center)

    mask = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
    cv2.circle(mask,(center_y,center_x),r,(1,1,1),-1,8,0)

    cropped = red*mask
    where=np.where(cropped>0)
    dif = np.max(cropped[where])-np.min(cropped[where])
    cropped_normal = (red-np.min(cropped[where]))*(255/dif)*mask
    cv2.imwrite(out_path+ser+'-crop_norm-'+img_name,cropped_normal)
    threshold = 150
    while True:
        if threshold > 255:
            print('Cannot find OD')
            break
        binary_img = (cropped_normal>=threshold).astype('uint8')*255
        #cv2.imwrite(out_path+ser+'-bp'+str(threshold)+'-'+img_name,binary_img)
        images, contours, hierachy = cv2.findContours(binary_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)
        if len(contours) == 0:
            threshold = threshold+10
            continue
        best_cnt = contours[0]
        best_area = 0
        for cnt, hier in zip(contours, hierachy[0]):
            temp_area = cv2.contourArea(cnt)
            if hier[3]==-1 and temp_area>best_area:
                best_cnt = cnt
                best_area = temp_area
        if best_cnt.shape[0] < 10: #at least contour shout have 10 points
            threshold = threshold+10
            continue
        (x,y), (MA, ma), angle = cv2.fitEllipse(best_cnt)
        print((x,y))
        #el_img =  cv2.ellipse(origin, ((x,y), (MA, ma), angle), (0, 0, 255), 1) #red
        fit_ellipse_area = math.pi * MA * ma/4
        print(MA)
        print(ma)
        if best_area > mask.sum()/4 or ma/MA > 1.5:
            threshold = threshold+10
            continue
        else:
            circularity = best_area/fit_ellipse_area
        #print(MA)
        #print(ma)
        #print(best_area)
        #print(fit_ellipse_area)
        print(circularity)
        #print(threshold)
        if 0.95 < circularity < 1.05:
            break
        threshold = threshold+10
    el_img =  cv2.ellipse(origin, ((x,y), (MA, ma), angle), (255, 255, 255), -1) #red
    cv2.imwrite(out_path+ser+'-fit_ell-'+img_name,el_img)
    od_mask = (el_img==255)
    od_mask = od_mask.astype(np.uint8)
    od = od_mask[:,:,0] * gray_img
    cv2.imwrite(out_path+ser+'-od-'+img_name,od)


    #cv2.drawContours(img, [best_cnt], 0, (255, 0, 0), 1) #blue
    #print(hierachy)
    #cv2.imwrite(out_path+ser+'-contour-'+img_name,img)
    #epsilon = cv2.arcLength(best_cnt, True) * 0.01
    #approx_poly = cv2.approxPolyDP(best_cnt, epsilon, True)
    #cv2.drawContours(origin2, [approx_poly], 0, (0, 255, 0), 1) #green
    #cv2.imwrite(out_path+ser+'-contour_approx-'+img_name,origin2)
    #ellipse = cv2.fitEllipse(approx_poly)
    #el_img =  cv2.ellipse(origin2, ellipse, (0, 0, 255), 1) #red
    #cv2.imwrite(out_path+ser+'-contour_approx-'+img_name,el_img)
    #cv2.imshow('image',img)
    #cv2.imshow('result',result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    print()
'''
def kaiser_filtering(k_filter, green_im):
    f = cv2.dft(green_im.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    f_shifted = np.fft.fftshift(f)
    #four=((f_shifted[:,:,0]**2 + f_shifted[:,:,1]**2)**(1/2))/255
    #cv2.imwrite('./fft.jpg',four)
    f_complex = f_shifted[:,:,0]*1j + f_shifted[:,:,1]
    f_filtered = k_filter * f_complex
    f_filtered_shifted = np.fft.fftshift(f_filtered)
    inv_img = np.fft.ifft2(f_filtered_shifted) # inverse F.T.
    filtered_img = np.abs(inv_img)
    return filtered_img

def center_detection(img_path, out_path):
    img_name = img_path.split('/')[-1]
    rgb_img = cv2.imread(img_path)
    ser = img_name.split('.')[0]
    cv2.imwrite(out_path+ser+'-1원본-'+img_name,rgb_img)
    #od box는 height 와 width의 공약수로
    od_box = int(rgb_img.shape[1]/20)
    blue, green, red = cv2.split(rgb_img)
    #cv2.imwrite(out_path+ser+'-2green-'+img_name,green)
    #cv2.imwrite(out_path+ser+'-3red-'+img_name,red)
    kaiser1 = np.kaiser(od_box,14)[:,None] # 1D hamming
    kaiser2 = np.kaiser(od_box,14)[:,None]
    kaiser2d = np.sqrt(np.dot(kaiser1, kaiser2.T)) # expand to 2D hamming
    k_filter = kaiser1
    best = 0
    for i in range(rgb_img.shape[1]-1):
        k_filter=np.hstack((k_filter,kaiser1))
    #cv2.imwrite('./od/aaa.jpg',k_filter*255)
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
    #print('best : '+str(y_index)+'~'+str(y_index+od_box-1))
    #print('Y-coordinate of Optic Disk Center : ',str(y_index+(od_box-1)/2))
    #cv2.imwrite(out_path+ser+'-best-'+img_name,filtered_img)
    #x_best = filtered_img
    #b,green_x,r = cv2.split(x_best)
    best = 0
    for i in range(int(rgb_img.shape[1]/od_box)):
        if j*od_box+od_box>rgb_img.shape[0]+1:
            break
        else:
            green_im=green[y_index:y_index+od_box,i*od_box:i*od_box+od_box]
        filtered_img = kaiser_filtering(kaiser2d, green_im)
        #print(str(j)+'->'+str(filtered_img.sum()))
        if np.max(filtered_img)>best:
                best = np.max(filtered_img)
                x_index = i*od_box
        #cv2.imwrite('./od/origin_y'+str(i)+'.jpg',green_im)
        #cv2.imwrite('./od/filtered_y'+str(i)+'.jpg',filtered_img)
    green_im=green[y_index:y_index+od_box:,x_index:x_index+od_box]
    beste_filtered_img = kaiser_filtering(kaiser2d, green_im)


    #print('best : '+str(x_index)+'~'+str(x_index+od_box-1))
    #print('X-coordinate of Optic Disk Center : ',str(x_index+(od_box-1)/2))
    #cv2.imwrite(out_path+ser+'-bestOD-'+img_name,filtered_img)
    center_x = int(x_index+od_box/2)
    center_y = int(y_index+od_box/2)
    center = cv2.line(rgb_img, (center_x,center_y),(center_x,center_y), (255,0,0), 5)
    #cv2.imwrite(out_path+ser+'-4center-'+img_name, center)
    return center_x, center_y

def od_cropping(img_path, out_path, center_x, center_y, r):
    rgb_img = cv2.imread(img_path)
    gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    blue, green, red = cv2.split(rgb_img)
    img_name = img_path.split('/')[-1]
    print(img_name)
    ser = img_name.split('.')[0]
    cv2.circle(rgb_img, (center_x,center_y), r, (0,0,0), 2)
    #cv2.imwrite(out_path+ser+'-center-'+img_name,center)
    mask = np.zeros((rgb_img.shape[0], rgb_img.shape[1]), dtype=np.uint8)
    cv2.circle(mask, (center_x,center_y), r, (1,1,1), -1, 8, 0)
    cv2.imwrite(out_path+ser+'-3red-'+img_name,red)
    '''
    red = cv2.GaussianBlur(red,(15,15),0)
    cv2.imwrite(out_path+ser+'-4redgaus-'+img_name,red)
    '''
    cropped = red*mask
    cond=np.where(cropped>0)
    dif = np.max(cropped[cond])-np.min(cropped[cond])
    cropped_normal = (red-np.min(cropped[cond]))*(255/dif)*mask
    cv2.imwrite(out_path+ser+'-5crop_norm-'+img_name,cropped_normal)
    threshold = 140
    #print(np.average(cropped_normal))
    while True:
        #print(threshold)
        if threshold > 255:
            #print('Cannot find OD')
            return (0,0)
        binary_img = (cropped_normal>=threshold).astype('uint8')*255
        #cv2.imwrite(out_path+ser+'-bp'+str(threshold)+'-'+img_name,binary_img)
        images, contours, hierachy = cv2.findContours(binary_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)
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
        if best_area > mask.sum()/4 or ma/MA>1.5:
            threshold = threshold+10
            continue
        else:
            circularity = best_area/fit_ellipse_area
        #print(circularity)
        if 0.95 < circularity < 1.05:
            break
        threshold = threshold+10
    #epsilon = cv2.arcLength(best_cnt, True) * 0.002
    #approx_poly = cv2.approxPolyDP(best_cnt, epsilon, True)
    #cv2.drawContours(rgb_img, [approx_poly], 0, (255, 0, 0), 1)
    cv2.imwrite(out_path+ser+'-6contour-'+img_name,rgb_img)
    el_img =  cv2.ellipse(rgb_img, ((x,y), (MA, ma), angle), (255, 255, 255), -1) #red
    print((x,y))
    cv2.imwrite(out_path+ser+'-fit_ell-'+img_name,el_img)
    od_mask = (el_img==255)
    od_mask = od_mask.astype(np.uint8)
    od = od_mask[:,:,0] * gray_img
    cv2.imwrite(out_path+ser+'-2od-'+img_name,od)
    return (x, y)
def od_segmentation(img_path, out_path):
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    rgb_img = cv2.imread(img_path)
    if rgb_img.shape[0]>2000:
        print('So Large')
        return (0,0)
    #algorithm1
    center_x, center_y = center_detection(img_path, out_path)
    #algorithm2
    r= 250
    (x, y) =od_cropping(img_path, out_path, center_x, center_y, r)
    return (x,y)
if __name__ == '__main__':
    src_path = './origin_data/all_files'
    out_path = './od5/'
    file_list = os.listdir(src_path)
    #print(file_list)
    od_angle_list = []
    od_realangle_list = []
    od_filename_list = []
    c=0
    for file in file_list:
        img_path = src_path+'/'+file
        rgb_img = cv2.imread(img_path)
        if rgb_img.shape[0]<2000:
            c=c+1
    print(c)
    '''
    for file in file_list:
        #optic_detect(img_path, out_path)
        img_path = src_path+'/'+file
        img_name = file
        ser = img_name.split('.')[0]
        (x, y) = od_segmentation(img_path, out_path)


        rgb_img = cv2.imread(img_path)
        if rgb_img.shape[0]>2000:
            continue
        cv2.line(rgb_img, (959, 647), (int(x), int(y)), (255, 0, 0), 5)
        #cv2.line(rgb_img, (int(x), 647), (int(x), int(y)), (255, 0, 0), 5)
        cv2.line(rgb_img, (0, 647), (rgb_img.shape[1]-1, 647), (255, 0, 0), 5)
        cv2.imwrite(out_path+ser+'-10angle-'+img_name,rgb_img)
        if not (x,y) == (0,0):
            if x-959.5<0:
                od_realangle_list.append(math.pi+math.atan((y-647.5)/(x-959.5)))
            else:
                od_realangle_list.append(math.atan((y-647.5)/(x-959.5)))
            od_angle_list.append(abs(math.atan((y-647.5)/(x-959.5))))
            od_filename_list.append(file)
            print(math.atan((y-647.5)/(x-959.5))*180/math.pi)
    print('Average Angle : ', sum(od_angle_list)/len(od_angle_list)*180/math.pi)
    print('Max Angle : ', max(od_angle_list)*180/math.pi)
    print('Max Name : ', od_filename_list[od_angle_list.index(max(od_angle_list))])
    print('Min Angle : ', min(od_angle_list)*180/math.pi)
    print('Min Name : ', od_filename_list[od_angle_list.index(min(od_angle_list))])
    #od_segmentation('./origin_data/all_files/vk068335.jpg', out_path)
    '''
