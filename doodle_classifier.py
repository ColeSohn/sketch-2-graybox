from logging import raiseExceptions
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from joblib import dump, load
import os
import numpy as np
import pdb
import cv2

#Load in Dataset
def read_doodles():
    label_list = []
    dir_string = os.path.join(os.path.dirname(__file__), 'data')
    labels = np.array([])
    label_cnt = 0
    data = np.array([])
    for file in os.listdir(os.fsencode(dir_string)):
        #Add label strings to list (given by .npy name)
        filename = os.fsdecode(file)
        label_list.append(filename.split(".")[0])
        
        category_data = np.load(os.path.join(dir_string, filename))[:10000, :]/255
        print("Data: ",filename.split(".")[0]," ",category_data.shape)

        labels = np.append(labels, np.ones((category_data.shape[0]))*label_cnt, axis=0)
        if(label_cnt==0):
            data = category_data
        else:
            data = np.concatenate((data,category_data), axis=0)
        label_cnt+=1
    # print(data.shape)
    return label_list, labels, data

def plot_data(title, label_list, labels, data):
    fig = plt.figure(figsize=(8., 8.))
    fig.suptitle(title, fontsize=16)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(6, 6),  # creates 2x2 grid of axes
                 axes_pad=0.5,  # pad between axes in inch.
                 )

    for ax, im, l in zip(grid, data, labels):
        im = np.resize(im, (28,28))
        ax.set_axis_off()
        ax.imshow(im, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(label_list[int(l)  ])
    plt.show()

if __name__ == "__main__":
    visualize = False
    label_list, labels, data = read_doodles()
    np.save("label_strings.npy",np.asarray(label_list))

    #Split Data
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.5, shuffle=True)
    print("Training Samples: ",X_train.shape)
    print("Testing Samples:", X_test.shape)
    #cv2.imshow("shape_ex", X_train[1].reshape((28,28)))
    #cv2.waitKey(0)

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001, probability=True)


    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    dump(clf, 'model.joblib') 


    #print(X_test[0])
    #X_test = np.load("instancers.npy")
    #print(X_test[0])

    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test)
    print(predicted)
    
    if(visualize):
        plot_data("Training Data: ",label_list, y_train, X_train)
        plot_data("Predictions: ",label_list, predicted, X_test)

    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )