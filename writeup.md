## Self Driving Car Nanodegree Project 4: Advanced Lane Lines

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/camera_calibration_test.png "Undistorted"
[image2]: ./output_images/distortion_correction.png "Road Transformed"
[image3]: ./output_images/perspective_transform.png "Perspective Transform"
[image4]: ./output_images/perspective_transform.png "Perspective Transform2"
[image5]: ./output_images/sobel_image.png "Sobel Edge Detection"
[image6]: ./output_images/s_channel_image.png "S Channel"
[image7]: ./output_images/l_channel_image.png "L Channel"
[image8]: ./output_images/binary_image.png "L Channel"
[image9]: ./output_images/curve_fitting_for_lines.png "Curve Fitting"
[image10]: ./output_images/curve_fitting_for_lines2.png "Curve Fitting"
[image11]: ./output_images/polygon_drawing.png "Polygon"
[image12]: ./output_images/polygon_drawing2.png "Polygon"
[image13]: ./output_images/hls_s_channel.png "HLS"
[image14]: ./output_images/lab_b_channel.png "Lab"
[video1]: ./project_video_out.avi "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

I provide this document as the write-up of the project. This document answers the points of the rubric and the pipeline I used for the project. All the codes of the pipeline described here can be found in 'project_code.ipynb' jupyter notebook. In addition, I used another 'ProcessVideo.py' to process video on bare python instead of jupyer notebook for performance reasons.

### Camera Calibration

The code for this step is contained in the code cells under the heading 'Camera Calibration' of the IPython notebook located in "./project_code.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Distortion Correction

The code of the part can be found in the cells following 'Distortion Correction' heading in the jupyter notebook. In this part I took the camera calibration parameters, that was calculated in the previous part using the `cv2.calibrateCamera()` function, and applied them to the actual camera images the `cv2.undistort()` function

Result of this step is demonstrated with the following picture:
![alt text][image2]

#### 2. Perspective Transform
The code of the part can be found in the cells following 'Perspective Transform' heading in the jupyter notebook. I hand picked the source coordinates as follows, and used the following code to get the destination points:

```python
    offset = 10
    dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                         [img_size[0]-offset, img_size[1]-offset], 
                                         [offset, img_size[1]-offset]])

    y_min = 445
    y_max = 640
    src=np.float32([(575,y_min),(713,y_min),(1300,y_max),(90,y_max)])
```

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

The result of this transfomr is demonstrated in the following picture
Result of this step is demonstrated with the following picture:
![alt text][image3]
![alt text][image4]

#### 3. Color Transforms and Binary Images
As the next step I used edge detection and thresholding of different color transforms to get a binary image.I used sobel edge detection to find the edges specifically on the x axis. I also used two more features from HLS color space. I converted the image to HLS color space, using 'cv2.cvtColor(img, cv2.COLOR_BGR2HLS)' function, and used L and S channels.

I noticed that S channel, although giving information on both lines, gives more information about the yellow lines on the sides; and L channel and sobel edge information give more information about the white lines.

The code of the part can be found in the cells following 'Color Transforms and Binary Images' heading in the jupyter notebook.

The following picture demonstrates the result of sobel edge detection
![alt text][image5]

The following picture demonstrates the result of S channel thresholding
![alt text][image6]

The following picture demonstrates the result of L channel thresholding
![alt text][image7]

The resulting image is as follows
![alt text][image8]

**Addenum:** After I worked with the test videos I noticed that S channel of the HLS color space produces so much unwanted artifacts that I could not filter out either with thresholding or with binary operations. I found the idea of using Lab colorspace from ![Jeremy Shannon's github page] (https://github.com/jeremy-shannon/). 

b Component of Lab colorspace gives more reliable information about the yellow lines, compared to the S component of HLS space.
So I replaced the S component with b component.

The following pictures demonstrate the difference between using S component and b component. The S component of HLS (the upper image) has unwanted bright areas whereas b component of Lab (the lower image) gives a much clearer line.
![alt text][image13]  ![alt text][image14] 

#### 4. Curve Fitting on Binary Image and Finding The Lane Lines
I used the window masking method for finding the lane lines. To do this I separated image vertically into search regions of 50 pixels each. In each window I used the histogram method to find local peaks. I used these peaks as the center points of the windows. 

In the next part I located the pixels that fall inside these windows and fitted a second order polynomial to these pixels for each lane line. After I fitted the lines I made sure there is a fit by plotting it on the image.

The code of the part and the next part can be found in the cells following 'Curve Fitting on Binary Image and Finding The Lane Lines' heading in the jupyter notebook.


The resulting images are as follows
![alt text][image9]
![alt text][image10]

#### 5. Calculation of the Curvature and Offset
After fitting a line on the pixels of lane line, I translated these values into real world distances. I did this following the assumptions:
* The distance between two lane lines is 3.7 meters
* Length of the lane line is 3 meters long each

These lead to the following assignments:
* The horizontal distance between lines are approximately 700 pixels so I used '3.7 / 700' for translation factor horizontally
* The total lines and spaces in the image (which is 720 pixel long vertically) are approximately 12-13 times the length of a dashed line. So I used '40/720' for translation factor vertically.

![alt text][image3]

Using these translation factors, I used the fitted values, and used the formula for the curvature radius.  I calculated the radius of the curvature using these numbers.

These calculations gave me two lane curvatures. Here, I made an assumption that the line created with more pixels is more reliable. So, instead of directly averaging the curvatures of the two lanes, I used a weighted average, weights being the number of pixels that are used to calculate the curvature.

After that, I used the difference of the mid-point of the screen and the mid point of the lines. I used the first elements of the window centroids to calculate the difference between the line. Finally, I used the translation parameters to translate this offset in number of pixels to the real world values.

I used 'cv2.putText()' function to plot these values the image.

#### 6. Plotting of the Polygon onto the image

I calculated an inverse transform by using the source and destination points used in part two. However this time I changed the order so that the transform is inversed.

I used the line fitting values that I found in the part four and for each y on the image I calculated an x, for both of the lines. After that, I used 'cv2.fillPoly()' function to plot a polygon.

Finally I used the inverse transform to transform these polygons back from the warped domain to the image domain.

The resulting images are as follows
![alt text][image11]
![alt text][image12]


#### 7. Video Processing and Final Result

For performance reasons, I left jupyter notebook and implemented this same pipe line to 'ProcessVideo.py'. In this script, I open the video with OpenCv, process it with the pipe, and then write it to the output video file using OpenCv. I did not optimize the running performance but I used parallel processing of the frames in order to reduce the run time.


### Pipeline (video)



Here's a [link to my video result](./project_video_out.avi)
![alt text][video1]

I also added a [debug level video](./project_video_out_verbose.avi) which as four additional images on top of final image demonstrating intermediary steps of the process. I used this video to see the levels and fine tune the thresholds of binarization operation. The four binary images are the binarized-combined image, S channel, L channel, and Sobel edges from left to right.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most important pitfall of this pipeline is that the asphalt with different colors change the response of the binarization part of the pipeline. Maybe a normalization according to a center color or histogram equalization/ local histogram processing can increase the performance.

There is one more improvement can be made on this pipe. The course suggests that the algoritm can remember the previous lane line calculations and limit the search for the current frame. This will be very useful since we know that the curvature of the road cannot change discretely. When implented, this would increase the accuracy in the frames where binarization provides an inaccurate result. Limiting search space will also increase the speed.

Also I haven't implented sanity checks but they would be very useful in the frames with inadequate information. When the result is not sane the algoritm can ignore it and use the previous result. 
