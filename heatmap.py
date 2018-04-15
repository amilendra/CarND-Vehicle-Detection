import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label

# Read in a pickle file with bboxes saved
# Each item in the "all_bboxes" list will contain a 
# list of boxes for one of the images shown above
#box_list = pickle.load( open( "bbox_pickle.p", "rb" ))
box_list = [((800, 400), (900, 500)), ((850, 400), (950, 500)), ((1050, 400), (1150, 500)), 
            ((1100, 400), (1200, 500)), ((1150, 400), (1250, 500)), ((875, 400), (925, 450)), 
            ((1075, 400), (1125, 450)), ((825, 425), (875, 475)), ((814, 400), (889, 475)), 
            ((851, 400), (926, 475)), ((1073, 400), (1148, 475)), ((1147, 437), (1222, 512)), 
            ((1184, 437), (1259, 512)), ((400, 400), (500, 500))]

box_list = [((1184, 96), (1248, 160)), ((800, 160), (864, 224)), ((832, 160), (896, 224)), ((864, 160), (928, 224)), ((896, 160), (960, 224)), ((1024, 160), (1088, 224)), ((768, 192), (832, 256)), ((832, 192), (896, 256)), ((1056, 192), (1120, 256)), ((1184, 192), (1248, 256)), ((768, 224), (832, 288)), ((800, 224), (864, 288)), ((832, 224), (896, 288)), ((864, 224), (928, 288)), ((896, 224), (960, 288)), ((960, 224), (1024, 288)), ((992, 224), (1056, 288)), ((1088, 224), (1152, 288)), ((1120, 224), (1184, 288)), ((1152, 224), (1216, 288)), ((736, 256), (800, 320)), ((832, 256), (896, 320)), ((928, 256), (992, 320)), ((960, 256), (1024, 320)), ((1056, 256), (1120, 320)), ((1088, 256), (1152, 320)), ((736, 288), (800, 352)), ((768, 288), (832, 352)), ((800, 288), (864, 352)), ((832, 288), (896, 352)), ((1088, 288), (1152, 352)), ((1184, 288), (1248, 352)), ((736, 320), (800, 384)), ((960, 320), (1024, 384)), ((768, 352), (832, 416)), ((864, 352), (928, 416)), ((896, 352), (960, 416)), ((672, 384), (736, 448)), ((704, 384), (768, 448)), ((768, 384), (832, 448)), ((800, 384), (864, 448)), ((832, 384), (896, 448)), ((864, 384), (928, 448)), ((896, 384), (960, 448)), ((928, 384), (992, 448)), ((160, 416), (224, 480)), ((192, 416), (256, 480)), ((672, 416), (736, 480)), ((736, 416), (800, 480)), ((768, 416), (832, 480)), ((800, 416), (864, 480)), ((960, 416), (1024, 480)), ((0, 448), (64, 512)), ((32, 448), (96, 512)), ((64, 448), (128, 512)), ((96, 448), (160, 512)), ((224, 448), (288, 512)), ((256, 448), (320, 512)), ((640, 448), (704, 512)), ((832, 448), (896, 512)), ((864, 448), (928, 512)), ((928, 448), (992, 512)), ((960, 448), (1024, 512)), ((992, 448), (1056, 512)), ((288, 480), (352, 544)), ((320, 480), (384, 544)), ((480, 480), (544, 544)), ((576, 480), (640, 544)), ((608, 480), (672, 544)), ((640, 480), (704, 544)), ((736, 480), (800, 544)), ((768, 480), (832, 544)), ((1152, 480), (1216, 544)), ((256, 512), (320, 576)), ((288, 512), (352, 576)), ((320, 512), (384, 576)), ((1152, 512), (1216, 576)), ((1184, 512), (1248, 576)), ((480, 544), (544, 608)), ((576, 544), (640, 608)), ((640, 544), (704, 608)), ((928, 544), (992, 608)), ((960, 544), (1024, 608)), ((1056, 544), (1120, 608)), ((1152, 544), (1216, 608)), ((1184, 544), (1248, 608)), ((832, 576), (896, 640)), ((1184, 576), (1248, 640)), ((736, 608), (800, 672)), ((832, 608), (896, 672)), ((864, 608), (928, 672)), ((928, 608), (992, 672)), ((768, 640), (832, 704)), ((800, 640), (864, 704)), ((832, 640), (896, 704))]

# Read in image similar to one shown above 
image = mpimg.imread('test_images/test1.jpg')
heat = np.zeros_like(image[:,:,0]).astype(np.float)

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

# Add heat to each box in box list
heat = add_heat(heat,box_list)
    
# Apply threshold to help remove false positives
heat = apply_threshold(heat,1)

# Visualize the heatmap when displaying    
heatmap = np.clip(heat, 0, 255)

# Find final boxes from heatmap using label function
labels = label(heatmap)
draw_img = draw_labeled_bboxes(np.copy(image), labels)

fig = plt.figure()
plt.subplot(121)
plt.imshow(draw_img)
plt.title('Car Positions')
plt.subplot(122)
plt.imshow(heatmap, cmap='hot')
plt.title('Heat Map')
fig.tight_layout()
plt.show()