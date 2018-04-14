import os
import numpy as np
import cv2
from skimage.feature import hog

# Define a function to return HOG features and visualization
# Features will always be the first element of the return
# Image data will be returned as the second element if visualize= True
# Otherwise there is no second return element

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=True, 
                     feature_vec=True):
                         
    # TODO: Complete the function body and returns
    hog_features, hog_image = hog(img, orientations=orient,
                              pixels_per_cell=(pix_per_cell, pix_per_cell), 
                              cells_per_block=(cell_per_block, cell_per_block), 
                              visualise=vis, feature_vector=feature_vec,
                              block_norm="L2-Hys")
    return hog_features, hog_image

orientations = [6, 9, 12]
pixels_per_cell = [2, 4, 8, 16]
cells_per_block = [2, 4, 8]
folder_list = ['vehicles/GTI_Far/', 'vehicles/GTI_Left/',
               'vehicles/GTI_MiddleClose/', 'vehicles/GTI_Right/', 
               'vehicles/KITTI_extracted/',
               'non-vehicles/Extras/', 'non-vehicles/GTI/']
image_file_list = []
for folder in folder_list:
    print(folder)
    images = os.listdir(folder)
    image_file_list.extend([os.path.join(folder, x) for x in images if x.endswith('.png')])

print(image_file_list)
for image_file in image_file_list:
    #print(image_file)
    # Read in the image
    image = cv2.imread(image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Call our function with vis=True to see an image output
    features, hog_image = get_hog_features(gray, orient= 9, 
                            pix_per_cell= 8, cell_per_block= 2, 
                            vis=True, feature_vec=False)
    cv2.imwrite('output_images/' + str(image_file).replace('/', '_'), hog_image)
    
cv2.imshow("Example Car Image", image)
cv2.waitKey()
cv2.imshow('HOG Visualization', hog_image)
cv2.waitKey()
cv2.destroyAllWindows()