##PAGE Scaning FROM: https://dontrepeatyourself.org/post/learn-opencv-by-building-a-document-scanner/
##Edge contour help from: https://stackoverflow.com/questions/69369615/opencv-remove-doubled-contours-on-outlines-of-shapes-without-using-retr-externa

from imutils.perspective import four_point_transform
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import ImageGrid

#Resize given aspect ratio of img and given width or height (not both)
def asp_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

#Get img from path
def img_input(img_pth, width=1000):
    image = cv2.imread(img_pth)
    image = asp_resize(image, width)
    return image

#Project page
def warp_page(img):
    orig_image = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert the image to gray scale
    blur = cv2.GaussianBlur(gray, (5, 5), 0) # Add Gaussian blur
    edged = cv2.Canny(blur, 75, 200) # Apply the Canny algorithm to find the edges

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cv2.drawContours(img, contours, -1, (255, 0, 0), 5)
    cv2.waitKey(0) 

    # go through each contour
    for contour in contours:
        #approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
        #print(len(approx))
        # if we found a countour with 4 points we break the for loop
        # (we can assume that we have found our document)
        if len(approx) == 4:
            # new_img = orig_image.copy()
            # cv2.drawContours(new_img, [approx], -1, (0, 255, 0), 3)
            # cv2.imshow("bbox", asp_resize(new_img, width=540))
            # cv2.waitKey(0) 
            
            doc_cnts = approx
            break

    orig_image_2 = orig_image.copy()

    # We draw the contours on the original image not the modified one
    cv2.drawContours(orig_image, [doc_cnts], -1, (0, 255, 0), 3)

    # apply warp perspective to get the top-down view
    warped = four_point_transform(orig_image_2, doc_cnts.reshape(4, 2))

    # convert the warped image to grayscale
    return warped, img

def resize_padded_square(img, size=28, color=[0,0,0]):
    max_ax = np.argmax(img.shape)
    assert(max_ax==1 or max_ax==0)
    if(max_ax==0):
        img=asp_resize(img, height=size)
        dw = size-img.shape[1]
        left, right = dw//2, dw-(dw//2)
        img = cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT,None, color)
    else:
        img=asp_resize(img, width=size)
        dh = size-img.shape[0]
        top, bottom = dh//2, dh-(dh//2)
        img = cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT,None, color)
    return img


def plot_imgs(title, imgs, titles, colors, color_key=False):
    size = 1000
    fig = plt.figure(figsize=(10., 5.))
    fig.suptitle(title, fontsize=16)
    grid = ImageGrid(fig, 111, 
                 nrows_ncols=(1, len(imgs)),  # creates 2x2 grid of axes
                 axes_pad=0,  # pad between axes in inch.
                 )
    for ax, im, l, c in zip(grid, imgs, titles, colors):
        im = resize_padded_square(im, size, color=[255, 255, 255])
        ax.set_axis_off()
        if(color_key and l=="Output"):
            red_patch = mpatches.Patch(color='red', label='Instancers')
            blue_patch = mpatches.Patch(color='blue', label='Wall Geo')
            green_patch = mpatches.Patch(color='green', label='Floor Geo')
            ax.legend(handles=[red_patch, blue_patch, green_patch], loc='lower center', fontsize="x-small", ncol=3)
        if c:
            ax.imshow(im)
        else: 
            ax.imshow(im,cmap='gray')
        ax.set_title(l)
    plt.show()

#Crop warped img in center (removes page edge artifacts that would lead to additional contours)
def center_crop(img, ratio):
    w = img.shape[1]*ratio
    h = img.shape[0]*ratio
    center = [img.shape[0] / 2, img.shape[1] / 2]
    x = center[1] - w/2
    y = center[0] - h/2
    crop = img[int(y):int(y+h), int(x):int(x+w)]
    return crop