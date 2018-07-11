# Project 05 - Vehicle Detection

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Normalize the features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.



## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### code overview:

`search_classify_main.py` - Main file for running all parts of this project.
`search_classify_hlpr.py` - Most of the helper functions used by pipeline.
`falsePos_and_MultDet_filter.py` - Functions for filtering false positives and combining multiple detections.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for extracting hog features is located in the main file `search_classify_main.py` followed by the comment `WRITEUP1` for the 'car' images and once again after `WRITEUP2`  for 'non-car' images.

In order to choose the best color space and channel\s, I first wrote some code to visualize the images in different colior spaces.

Comparing an image between all color spaces:

![](writeup_images\all_color_channels.png)

This comparison convinced me that each color space might have noisier channels than others. Noisy images would produce noisy gradients in the hog features. So in order to find the most stable (not noisy) channel\s I also plotted each channel for each color space. When going over them for a number of images, I was convinced that channel 0 of **YCrCb** is a stable option consistently yielding smooth images while channel:

![](C:\Users\ROEE\GitProjects\ObjectDetection\ObjDet\writeup_images\YCrCb_channels.png)

I grabbed random images and displayed them to get a feel for what the `skimage.hog()` output looks like. Here is an example plotting the different color spaces and HOG parameters of `orientations=9, ` `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![1531208358150](C:\Users\ROEE\GitProjects\ObjectDetection\ObjDet\writeup_images\hogImages)

Here I noticed that `hog(YCrCb[:,:,0])` (top middle) did produce nice "car-like" images as I thought was needed, but I also noticed that in `hog(YCrCb[:,:,1])`  (bottom middle) there is also "car-like" image noticeable by its "circling" of the break lights of the vehicle. So if extracting features from a single channel would produce bad results I noted to myself that adding more channels would probably help.

2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. I did find that adding channels to the hog features produces better results in the final video classification task but not without a cost. Increasing the number of channels also reduces the speed of the running code (on my laptop) by a factor of 3-4. The algorithm demands of reliability vs real-time capabilities will determine how many channels to choose. I provided a video of both options.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM based on the following steps (in the main file after comment `WRITEUP3`):

1. Extract features from images:

   1. Convert image to YCrCb.

   2. Extract spatial features from a scaled down version of the image (16, 16, 3).

      (16x16x3 = 768 features)

   3. Extract from the regular size image a color histogram of 16 bins per channel (x3).

      (48 features)

   4. Extract hog features from the image with:

     ```
     orient = 9  # HOG orientations
     pix_per_cell = 8  # HOG pixels per cell
     cell_per_block = 2  # HOG cells per block
     hog_channel = "ALL"
     ```
     which means (9 orientations)x(64//8 -1)x(64//8 -1)x2x2x3 = (1764 features per channel) x3 = 5,292 features.

   This sums up to a total of 6,108 features per picture. Note that reducing HOG features from 3channels to 1, cuts down on 58% of the features.

2. Use `train_test_split` to split the data into a training set and a test set.

3. Obtain a scaler based on the training set (with `StandardScaler` from sklearn.preprocessing).

4. Fit the data with a linear SVC.

5. Store the model and model parameters using pickle.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search window positions as seen in the following image:

![](writeup_images\searchAreaAndGrid.png)

The idea is simple, small cars are far away and therefore usage of small window sizes (of 64x64) are only near the horizon ( $y \in [400,500]$ ) and vise versa. For illustration purposes the image above includes only 25% of the searched windows otherwise the overlapping of the windows would "clog" the image. I chose the `xy_overlap` to be 70% because the training set images are centralized (cars appear in the middle of image). Reducing `xy_overlap` would result in images where the car is not in the center and therefore bad classification results.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales (128, 96 and 64) using YCrCb 3-channel HOG features plus spatially binned scaled down to 16x16, 3 channel YCrCb and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

<img src="output_images\test1_scvModel_spatSz16_hogAll_allBBox.png" style="zoom:80%" />

In the image above, I plotted all the bounding boxes found by the svc with the sliding windows.

In order to obtain the final bounding boxes, I later applied a heat map image, where every bounding box adds 1 to every pixel inside its bounds. Then I applied a threshold of >1 to the image, to get rid of false positive classifications. On the thresh hold image I used `scipy.ndimage.measurements.label()`to identify separate "blobs" in the image. Then for each label, I found the corresponding bounding box. This process of obtaining the heatmap is shown in the following image:

![](output_images\test1_scvModel_spatSz16_hogAll_heatMapFiltered.png)

Another example:

<img src="output_images\test6_scvModel_spatSz16_hogAll_allBBox.png" style="zoom:80%" />


![](output_images\test6_heatMapFiltered.png)



### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](test_video_full_scvModel_spatSz16_hogAll_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The only difference between the video pipeline and the stills pipeline is that in the video pipe I used the bounding boxes, found by every pair of sequential frames as input to the heatmap and raised the threshold to >2. This produced good results as can be seen in the video.



### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

