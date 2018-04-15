import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import time
import random
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

### TODO: Tweak these parameters and see how the results change.
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb(HLS bit good)
orient = 12  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 1 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
x_start_stop = [320, 1280] # Min and max in y to search in slide_window()
y_start_stop = [400, 656] # Min and max in y to search in slide_window()
scale = 1.0


# NOTE: the next import is only valid for scikit-learn version <= 0.17
#from sklearn.cross_validation import train_test_split
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        orig_img = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
        cv2.imshow("orig_img", orig_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        cv2.imshow("test_img", test_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            print("Matched", window)
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

# Read in cars and notcars
#images = glob.glob('*.jpeg')
folder_list = ['vehicles/KITTI_extracted/',
               'non-vehicles/Extras/']
folder_list = ['vehicles/GTI_Far/', 'vehicles/GTI_Left/',
               'vehicles/GTI_MiddleClose/', 'vehicles/GTI_Right/', 
               'vehicles/KITTI_extracted/',
               'non-vehicles/Extras/', 'non-vehicles/GTI/']
folder_list = ['vehicles_smallset/cars1/', 'vehicles_smallset/cars2/', 'vehicles_smallset/cars3/',
               'non-vehicles_smallset/notcars1/', 'non-vehicles_smallset/notcars2/', 'non-vehicles_smallset/notcars3/']

images = []
for folder in folder_list:
    print(folder)
    contents = os.listdir(folder)
    images.extend([os.path.join(folder, x) for x in contents if x.endswith('.jpeg')])

cars = []
notcars = []
for image in images:
    if 'non-vehicles' in image:
        notcars.append(image)
    else:
        cars.append(image)

#print(cars)
#print(notcars)
# Reduce the sample size because
# The quiz evaluator times out after 13s of CPU time
#random.shuffle(cars)
#random.shuffle(notcars)
sample_size = 500
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]

car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
#print(car_features)
#print(notcar_features)
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)
#X_scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X)
#scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)
    
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# svc = LinearSVC(C=1.0, class_weight=None, dual=True, 
#                 fit_intercept=True, intercept_scaling=1, loss='squared_hinge', 
#                 max_iter=1000, multi_class='ovr', penalty='l2', random_state=None, 
#                 tol=0.0001, verbose=0)
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()

i = 0
def process_image(image):
    global i
    #out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    #cv2.imwrite('input_images/test%d.jpg' % (i),image)
    #i = i + 1
    #return image
    #image = mpimg.imread('test_images/test6.jpg')
    draw_image = np.copy(image)

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    #image = image.astype(np.float32)/255

    box_list = []
    #j = 0
    #for xy_window in [(32, 32)]:#, (64, 64), (96, 96), (128, 128)
        #j = j + 1
    windows = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop, 
                        xy_window=(256, 256), xy_overlap=(0.5, 0.5))

    search_grid_img = draw_boxes(draw_image, windows, color=(255, 0, 0), thick=0) 
    cv2.imshow("Grid", search_grid_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    canditates = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)

    box_list.extend(canditates)
    #window_img = draw_boxes(draw_image, canditates, color=(0, 0, 255), thick=j)                    
    #print(box_list)
    #cv2.imwrite('pipe_images/window_img%d.jpg' % (i),window_img)

    #find_cars_img = find_cars(image, y_start_stop[0], y_start_stop[1], 
    #                          scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    #cv2.imwrite('pipe_images/find_cars_img%d.jpg' % (i),find_cars_img)
    #plt.imshow(window_img)
    #plt.show()

    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,box_list)
        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    cv2.imwrite('pipe_images/test%d.jpg' % (i),draw_img)
    i = i + 1
    return draw_img

#img = cv2.imread('test_images/test6.jpg')
#curved = process_image(img)
#cv2.imshow("output_images/with_radius_test3.jpg", curved)
#cv2.waitKey()
#cv2.destroyAllWindows()

test_images = [
    # 'cutouts/bbox-example-image.jpg',
    # 'test_images/test1.jpg',
    # 'test_images/test2.jpg',
    # 'test_images/test3.jpg',
    # 'test_images/test4.jpg',
    # 'test_images/test5.jpg',
    # 'test_images/test6.jpg',
    #'input_images/test215.jpg',
    'input_images/test304.jpg',
    # 'input_images/test323.jpg',
    # 'input_images/test466.jpg',
    # 'input_images/test989.jpg',
]

for img in test_images:
    image = cv2.imread(img)
    result = process_image(image)
    cv2.imshow(img, result)
    cv2.waitKey()
    cv2.destroyAllWindows()

# white_output = 'test_video_output.mp4'
# clip1 = VideoFileClip("test_video.mp4")
# white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# white_clip.write_videofile(white_output, audio=False)

# white_output = 'project_video_output.mp4'
# clip1 = VideoFileClip("project_video.mp4")
# white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# white_clip.write_videofile(white_output, audio=False)

