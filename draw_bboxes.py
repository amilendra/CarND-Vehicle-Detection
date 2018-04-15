import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('cutouts/bbox-example-image.jpg')

# Define a function that takes an image, a list of bounding boxes, 
# and optional color tuple and line thickness as inputs
# then draws boxes in that color on the output

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on your image copy using cv2.rectangle()
    # return the image copy with boxes drawn
    for bbox in bboxes:
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    return draw_img # Change this line to return image copy with boxes
# Add bounding boxes in this format, these are just example coordinates.
bboxes = [((6, 492), (103, 515)), 
          ((257, 497), (298, 512)),
          ((281, 505), (371, 563)),
          ((842, 500), (1114, 669)),
          ((482, 508), (532, 555)),
          ((594, 512), (634, 546)),]

result = draw_boxes(image, bboxes, thick=2)
cv2.imshow("Result", result)
cv2.waitKey()
cv2.destroyAllWindows()

