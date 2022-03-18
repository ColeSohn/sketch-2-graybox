##PAGE Scaning FROM: https://dontrepeatyourself.org/post/learn-opencv-by-building-a-document-scanner/
##Edge contour help from: https://stackoverflow.com/questions/69369615/opencv-remove-doubled-contours-on-outlines-of-shapes-without-using-retr-externa

import cv2
import numpy as np
import importlib
import s2g_imtools as s2g_im
importlib.reload(s2g_im)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from joblib import dump, load

import pdb

#Input: img: scanned (projected) img of drawing on paper
#Output: contours, outer_contours, and imgs for visualization
def extract_polygons(img):
    blank = np.zeros_like(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (3, 3), 0) # Add Gaussian blur
    edged = cv2.Canny(blur, 75, 200) # Apply the Canny algorithm to find the edges

    #Wall Contours
    cs, _ = cv2.findContours(edged,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if(len(cs)==0):
        return [], [], img, blank
    cont_img = img.copy()
    cv2.drawContours(cont_img, cs, -1, (255, 255, 255), 3)
    blur2 = cv2.GaussianBlur(edged, (11, 11), 0) # Add Gaussian blur
    cs2, h = cv2.findContours(blur2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    outer_conts = []
    inner_conts = []
    conts = []
    for c, hrc in zip(cs2, h[0]):
        # we approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.0001 * peri, True)
        conts.append(approx)
        if(hrc[2]<0):
            inner_conts.append(approx)
        elif(hrc[3]<0):
            outer_conts.append(approx)
    

    cv2.drawContours(blank, outer_conts, -1,(255,255,255), 3)
    return cs, outer_conts, cont_img, blank

def split_drawings(img):
    #wb = cv2.xphoto.createSimpleWB()
    #wb.setP(0.2)
    #img = wb.balanceWhite(img)
#     cv2.imshow("wb",s2g_im.asp_resize(img, width=540))
#     cv2.waitKey(0) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
 
    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 80, 20])
    upper1 = np.array([10, 255, 255])
 
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160,80,20])
    upper2 = np.array([179,255,255])
 
    lower_mask = cv2.inRange(img, lower1, upper1)
    upper_mask = cv2.inRange(img, lower2, upper2)
    img_red = lower_mask + upper_mask

    # lower_black = np.array([0,0,0])
    # upper_black = np.array([180,90,150])
    # img_black = cv2.inRange(img, lower_black, upper_black)

    lower_blue = np.array([70,0,0])
    upper_blue = np.array([150,255,255])
    img_blue = cv2.inRange(img, lower_blue, upper_blue)
    #Show
    # cv2.imshow("thresh",s2g_im.asp_resize(img_red, width=540))
    # cv2.waitKey(0) 
    # cv2.imshow("thresh", s2g_im.asp_resize(img_black, width=540))
    # cv2.waitKey(0) 
    return (img_blue, img_red)

#https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
def intersection_over_union(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    boxBArea = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    merged = [min(a[0], b[0]),min(a[1], b[1]),max(a[2], b[2]), max(a[3], b[3])]
    return iou, merged


def merge(boxes):
    thresh = 0.1
    merged = []
    num_merged = 0
    for b in range(len(boxes)):
        numbmerged = 0
        for c in range(len(boxes)):
            iou, m = intersection_over_union(boxes[b], boxes[c])
            if(iou>thresh and iou<1):
                merged.append(m)
                numbmerged+=1
        num_merged += numbmerged
        if(numbmerged==0):
            merged.append(boxes[b])
    return merged, num_merged

def doodle_bounds(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    conts, _, _, _ = extract_polygons(img.copy())
    boxes = []
    vis_imgs = []

    #Initial bboxes
    length_thresh = 70 #For removing small contours
    img_bbox_1 = img.copy()
    for cnt in conts:
        if(cv2.arcLength(cnt,True)<length_thresh):
            continue
        x,y,w,h = cv2.boundingRect(cnt)
        img_bbox_1=cv2.rectangle(img_bbox_1,(x,y),(x+w,y+h),(0,0,255),2)
        boxes.append([x, y, x+w, y+h])
    new_boxes = boxes
    vis_imgs.append(img_bbox_1)

    cnt = 0
    while(cnt<2):
        cnt+=1
        new_boxes, num = merge(new_boxes)
        if(num==0):
            break
    
    img_bbox_2 = img.copy()
    res = []
    for i in new_boxes:
        if i not in res:
            res.append(i)
            img_bbox_2=cv2.rectangle(img_bbox_2,(i[0],i[1]),(i[2],i[3]),(0,255,0),2)
    new_boxes = res
    vis_imgs.append(img_bbox_2)

    return new_boxes, vis_imgs

def instancers(og_image, instancer_thresh):
    bboxes, vis = doodle_bounds(instancer_thresh)
    instancers = []

    for b in bboxes:
        inst_im = og_image[b[1]:b[3],b[0]:b[2]]
        inst_im = cv2.cvtColor(inst_im, cv2.COLOR_BGR2GRAY)
        ret,inst_im = cv2.threshold(inst_im,140,255,cv2.THRESH_BINARY_INV)
        inst_im = s2g_im.resize_padded_square(inst_im, 28)
        instancers.append(inst_im.flatten(order='C'))
    clf = load('G:/My Drive/Sketch2Graybox/model.joblib') 
    instancers = np.array(instancers)/255

    predicted_labels = []
    predicted_label_strings = []
    pred_imgs = instancers
    if(instancers.shape[0]>0):
        print(clf.predict_proba(instancers))
        predicted_labels = clf.predict(instancers)

        label_strings = np.load("G:\My Drive\Sketch2Graybox\label_strings.npy")
        predicted_label_strings = label_strings[predicted_labels.astype(int)]
    out = []
    for b, s in zip(bboxes, predicted_label_strings):
        out.append((b, s))

    return out, pred_imgs, vis

def plot_data(title, labels, data):
    fig = plt.figure(figsize=(8., 8.))
    fig.suptitle(title, fontsize=16)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(1, len(labels)),  # creates 2x2 grid of axes
                 axes_pad=0.5,  # pad between axes in inch.
                 )

    for ax, im, l in zip(grid, data, labels):
        im = np.resize(im, (28,28))
        ax.set_axis_off()
        ax.imshow(im, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(l)
    plt.show()

def vis_instancers(inst, img, pred_imgs):
    vis_pred_imgs = True
    #Bounding Boxes
    ls = []
    ims = []
    for i in inst:
        b = i[0]
        label = i[1]
        ls.append(label)
        img=cv2.rectangle(img,(b[0],b[1]),(b[2],b[3]),(0,0,255),2)
        cv2.putText(img, label, (b[0], b[1]), cv2.FONT_HERSHEY_DUPLEX , 1.0, [0, 0, 255], 1)
    #cv2.imshow("bbox", s2g_im.asp_resize(img, width=540))
    #cv2.waitKey(0)
    data = [ls, pred_imgs]
    return img, data

def contour_mask(img):
    conts, _, _, _ = extract_polygons(img)
    black = np.zeros_like(img, dtype="uint8")
    white = np.ones_like(img)*255
    cv2.drawContours(black, conts, -1,(255,255,255), 5)
    masked = cv2.bitwise_and(img, black)
    white = cv2.bitwise_and(white, cv2.bitwise_not(black))
    out = white+masked
    #cv2.imshow("masked", s2g_im.asp_resize(out, width=540))
    #cv2.waitKey(0) 
    return out

def vis_imgrid(title, imgs, labels, col_flags, color_key=False):
    assert(len(imgs)==len(labels) and len(imgs)==len(col_flags))
    for i in range(len(col_flags)):
        if(col_flags[i]==1):
            imgs[i]=cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
    s2g_im.plot_imgs(title, imgs,labels,col_flags, color_key=color_key)


#Function that takes img path and returns data for graybox generation
#Input: img path and optional visualize bool
#Output: tuple of level_bound_polys and instancer_data
def sketch_2_graybox_data(img_pth, vis_overview=False, vis_page_warp=False, vis_level_geo=False, vis_inst_bbox=True, vis_inst_clf=False):
    width = 1500
    img = s2g_im.img_input(img_pth, width) #Img array from file
    img_scanned, img_page_conts = s2g_im.warp_page(img.copy()) #"Scan" page
    img_scanned = s2g_im.center_crop(img_scanned, 0.97)  #Crop to center to remove page boundaries

    #Mask using contours
    contour_masked = contour_mask(img_scanned.copy())

    #Split level contours from instancer doodles by color
    geo_contour_thresh, instancer_thresh = split_drawings(contour_masked.copy())

    #Extract level_geo_contours
    wall_conts, outer_conts, img_all_conts, img_outer_conts = extract_polygons(cv2.cvtColor(geo_contour_thresh.copy(), cv2.COLOR_GRAY2RGB)) #Extract polygons from warped img
    geo_contours = (wall_conts, outer_conts)

    #Separate and classify instancers
    instancer_data, pred_imgs, inst_vis = instancers(img_scanned.copy(), instancer_thresh)
    
    #Visualizers
    if vis_overview:
        out = np.ones_like(img_scanned)*255
        out[np.where(img_all_conts == [255])] = [255, 0, 0]
        img_outer_conts_bw = cv2.cvtColor(img_outer_conts, cv2.COLOR_BGR2GRAY)
        out[np.where(img_outer_conts_bw == [255])] = [0, 255, 0]
        out[np.where(instancer_thresh == [255])] = [0, 0, 255]
        out, _ = vis_instancers(instancer_data, out, pred_imgs)
        pad = 2
        out = cv2.copyMakeBorder(out, pad, pad, pad, pad, cv2.BORDER_CONSTANT,None, [0,0,0])
        vis_imgrid("Interpretted Sketch", [img.copy(), out], ["Input", "Output"],[1,1], color_key=True)

    if vis_page_warp:
        vis_imgrid("Page Scanning", [img.copy(), img_page_conts.copy(), img_scanned.copy()], ["Input","Contours", "Warped"], [1, 1, 1])
    if vis_level_geo:
        vis_imgrid("Level Geo", [img_scanned.copy(),contour_masked.copy(),geo_contour_thresh.copy(), img_all_conts.copy(), img_outer_conts.copy()], ["Input","Contour Mask", "Threshold", "Contours", "Outer Contours"], [1, 1,0, 0, 0])
    if vis_inst_bbox:
        inst_vis = [instancer_thresh.copy()]+inst_vis
        vis_imgrid("Instancers", inst_vis, ["Threshold", "Contour Bboxes", "Merged Bboxes"], [1, 1, 1]) 
    if vis_inst_clf:
        inst_clf_vis, data = vis_instancers(instancer_data, img_scanned.copy(), pred_imgs)
        vis_imgrid("Instancer Predictions", [inst_clf_vis], [""], [1]) 
        plot_data("Predictions", data[0], data[1])
    return width, geo_contours, instancer_data

if __name__ == "__main__":
    sketch_2_graybox_data("G:/My Drive/Sketch2Graybox/drw_test/im11.jpg", vis_overview=True, vis_page_warp = False, vis_level_geo=False, vis_inst_bbox=False, vis_inst_clf=True)