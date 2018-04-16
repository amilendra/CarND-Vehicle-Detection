## Vehicle Detection Project Writeup

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2-CH1]: ./output_images/HOG_example-CH1.png
[image2-CH2]: ./output_images/HOG_example-CH2.png
[image2-CH3]: ./output_images/HOG_example-CH3.png
[searchgrid_1]: ./output_images/test0_searchgrid_1.jpg
[searchgrid_2]: ./output_images/test0_searchgrid_2.jpg
[searchgrid_3]: ./output_images/test0_searchgrid_3.jpg
[output_1]: ./output_images/test0_output.jpg
[output_2]: ./output_images/test3_output.jpg
[output_3]: ./output_images/test4_output.jpg
[output_4]: ./output_images/test5_output.jpg
[bboxes_and_heat_1]: ./output_images/bboxes_and_heat_1.png
[bboxes_and_heat_2]: ./output_images/bboxes_and_heat_2.png
[bboxes_and_heat_3]: ./output_images/bboxes_and_heat_3.png
[bboxes_and_heat_4]: ./output_images/bboxes_and_heat_4.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook `project.ipynb`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2-CH1]
![alt text][image2-CH2]
![alt text][image2-CH3]

#### 2. Explain how you settled on your final choice of HOG parameters.

When I first started work on this, I did some crude brute force testing and found out that RGB color space gave less than 0.6 accuracy when used without histogram bins or spacial bins. 

I then tried `YCrCb` with all permissible channels, and the results were promising, but for some reason `HLS` was slightly better.

I must admit that these were basically trial and error, and nothing was systematic. While changing color spaces and hog channels, I was changing the windowing mechanisms as explained later, so there may be better parameters than this. But did not have time to explore more and the video is reasonably accurate so stopped at this.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I mostly based my work on the functions provided by the lessons.
Main change I did was to refactor `extract_features` in `lesson_functions.py` so that it used `single_img_features` for its implementation. I did not like the code redundancy. 

I trained a linear SVM using the following parameters.  HOG features, Histogram features, spatial features, all were used. See lines 21 to 33 of `pipeline.py`.

```python
color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb(HLS bit good)
orient = 9  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```
Extracting features and the actual training is done between line 115 to 157 of `pipeline.py`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
At first, I lazily tried searching throughout the whole image without using the x_start_stop and y_start_stop parameters,
but this was taking too long. So I used the y_start_stop parameter to exclude the sky from the search area. This gave good results, but there were some false positives,
where part of middle white lane lines were errorneously being matched as vehicles.
Because vehicles only appear in the bottom-right quadrant of the image, I tuned x_start_stop so that the mid-line of the road is excluded from the search area.

One other trick I used was to use 3 window sizes:64x64, 96x96, 256x256.
This was done because depending on how far the vehicle is, sometimes the vehicle would not fit within the search window,
so vehicles were either not being detected at all, or they were being detected only when they appeared in a certain size.

Because the 256x256 is a fairly large window size, having more overlap between search grids may give better vehicle detection results, so I used a 75% overlap.
A 50% overlap would have been enough for smaller window sizes, but I did not want to go through the trouble of changing the overlap depending on the window size,
so I kept it 75% for all window sizes.

Here is a sample for the 64x64 search window.
![alt text][searchgrid_1]

Here is a sample for the 96x96 search window.
![alt text][searchgrid_2]

Here is a sample for the 256x256 search window.
![alt text][searchgrid_3]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using HSV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][output_1]
![alt text][output_2]
![alt text][output_3]
![alt text][output_4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. 
I did not use the suggested method of `scipy.ndimage.measurements.label()` to threshold over multiple frames.

### Here are four frames and their corresponding heatmaps:

![alt text][bboxes_and_heat_1]
![alt text][bboxes_and_heat_2]
![alt text][bboxes_and_heat_3]
![alt text][bboxes_and_heat_4]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

As explained previously, using search windows of varying scales helped a lot in improving accuracy. Restricting the image search area also helped reducing false positives, but I am not sure if that is practical in real life. Thankfully, there were no vehicles right in front of of the same lane, or there weren't any vechiles coming towards me from the left hand side of the road so this pipeline works only for this particluar case. A more general solution would be good to have.

Although it improves accuracy, iteratively searching through search windows of multiple sizes, makes the processing time of the pipeline very long. The whole video took abot 1 hour to process, which may not be good as a real time vehicle detection system.

Only fairly large vehicles are detected and vehicles that are faraway are not detected. Also I found that sometimes, the white car dissapears when it is passing the white colorish road. I also see intermittent false positives so one way of improving will be to use `scipy.ndimage.measurements.label()` over a series of frames. However I did not have time to attempt that.