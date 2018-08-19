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

[image0]: ./output_images/chessboard_corners.png "Chessboard Corners"
[image1]: ./output_images/distorted_image.png "Distorted"
[image2]: ./output_images/undistorted_image.png "Undistorted"
[image3]: ./output_images/pre_color_thresh.png "Before Color Threshold"
[image4]: ./output_images/post_color_thresh.png "After Color Threshold"
[image5]: ./output_images/original_with_source_and_dest.png "Source and Destination Points"
[image6]: ./output_images/warped_with_dest.png "Warped with Destination Points"
[image7]: ./output_images/pre_top_down.png "Pre Transform"
[image8]: ./output_images/post_top_down.png "Post Transform"
[image9]: ./output_images/pre_grad_thresh.png "Pre Gradient Threshold"
[image10]: ./output_images/post_grad_thresh.png "Post Gradient Threshold"
[image11]: ./output_images/original_histogram.png "Original"
[image12]: ./output_images/original_bin_histogram.png "Binary Image"
[image13]: ./output_images/histogram.png "Histogram"
[image14]: ./output_images/poly_hist.png "Histogram Search Method"
[image15]: ./output_images/poly_poly.png "Polynomial Search Method"

[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

My first step of this project was to calibrate the camera so that I may perform an undistort transformation and remove the effect that the camera lense may have on the image. Specifically, I needed to calculate the camera matrix and distortion coefficients. To do this I relied heavily on the cv2 library.

The camera calibration was performed in `calibrate_camera.py`. In this script, I loop through the 20 calibration images of a chessboard, find the corners with `cv2.findChessboardCorners()`, add the image points to a list (2D image points) and the corresponding object points (3D real world points). With these points stored, I passed them to `cv2.calibrateCamera()` in order to calculate the camera matrix and distortion coefficients. An example of the `cv2.findChessboardCorners()` can be seen here:

![alt text][image0]

These camera coefficient outputs could then be used with `cv2.undistort()` in order to correct an image. See the distorted and undistorted example below:

![alt text][image1]
![alt text][image2]

### Pipeline (single images)

#### 1. Camera Undistort

The first operation in my final pipeline is to apply a camera undistortion. See the Camera Calibration section above for an example of how camera undistortion is applied. In the final pipeline, the camera undistortion was implemented in `locate_lane_lines.py` in the function `undistort()`.

#### 2. Color Detection

Next I applied a color threshold to identify white and yellow pixels in the original RGB image. After much testing, I determined that the best method to do so was to convert the RGB image to HSV color space. The H (hue) layer made it easier to identify specific colors (yellow and white) of the colors of lane lines. With a specific hue band targeted, I also applied a band of S (saturation) and V (value) thresholds to allow for image variance in the white and yellow targets. This algorithm was implemented in `locate_lane_lines.py` in the function `color_thresh()`.

An example of how this color thresholding works may be found here:

![alt text][image3]
![alt text][image4]

#### 3. Top Down Perspective Transform

Next I performed a top down perspective transform. In order to achieve this, I manually selected the source and destination points on a sample image. Those points can be seen here:

![alt text][image5]

The red points/lines are the source points (points to be mapped to a new perspective) and the blue points/lines are the destination points (points to be mapped to in the new perspective). Those points are listed here:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 709, 463      | 1034, 0       | 
| 1034, 664     | 1034, 720     |
| 274, 664      | 274, 720      |
| 573, 463      | 274, 0        |

With these points, I used the function `cv2.getPerspectiveTransform()` to get the transform and inverse transform matrices. Those matrices can be used with `cv2.warpPerspective()` in order to obstain a top down view perspective:

![alt text][image6]

The code to calculate these transofmation matrices may be found in `perspective_transform.py` and the implementation in the pipeline can be seen in `locate_lane_lines.py` in the function `top_down_xfrm()`. An example transform on a curved road can be seen here:

![alt text][image7]
![alt text][image8]

#### 4. Gradient Threshold

Next I applied a gradient threshold in the x (horizontal) direction in order to help detect the lane lines. I specifically chose to apply the gradient threshold after the top down transform because its clear that the lane lines will be very close to vertical in this perspective. This makes a sobel x gradient very appropriate. I used a gradient threshold of 10 to 100 and a default filter size of 1. I implemented this algorithm in `locate_lane_lines.py` in the function `grad_thresh()`. The effects of a x gradient threshold on a top down transformed image can be seen here:

![alt text][image9]
![alt text][image10]

#### 5. Lane Line Search and Polynomial Fit

The remainder of my pipeline was implemented in a class `LaneLines` so that important parameters could be efficiently stored and filtered.

The next step was to search for "activated" pixels (those pixels identified by the thresholding techniques) and to fit a 2nd order polynomial to those pixels. I utilized two separate methods to search for activated pixels of a given binary image:
1. Histogram Search
2. Polynomial Search

The first method, Histogram Search, was implemented in `locate_lane_lines.py` in the method `LaneLines.find_lane_pixels_hist()`. This method calculates a histogram of activated pixels on the bottom half of the input binary image and starts a search at each peak of the histogram. A sample histogram can be seen here:

![alt text][image11]
![alt text][image12]
![alt text][image13]

Subsequent pixels are found with a moving window that identifies the average location of activated pixels and moves the window accordingly. Various hyperparameters are used to calibrate this algorithm and can be found in the constructor method of `LaneLines`. Once the activated pixels are selected, they can be fed to `numpy.polyfit()` to calculate the coefficients and interpolated values. Here is an annotated image to show the sliding window (green box), found activated pixels (red dots), and resultant 2nd order polynmial fit (yellow line).

![alt text][image14]

The second method, Polynomial Search, was implemented in `locate_lane_lines.py` in the method `LaneLines.find_lane_pixels_poly()`. This method uses a previous polynomial fit as a starting point to search for activated pixels given some search window. This is a more light weight version of the histogram search method and is therefore used when a polynomial fit has already been found with some reasonable degree of certainty. In general, the lane lines will not change much from frame to frame so this is a reliable search method. An example of this search method can be seen here:

![alt text][image15]

#### 6. Real World Curvature and Vehicle Offset Calculations

With a polynomial expression of each lane line, it is possible to calculate important real world parameters that will be required for vehicle steering control. Namely, the radius of curvature of each lane line and the offset of the vehicle from the center of the lane. These algorithms are implemented in `LaneLines.calc_rad_real()` and `LaneLines.calc_offset()` respectively.

The equation for a 2nd order polynomial is f(y) = Ay^2 + By + C. The equation for the radius of curvature of this polynomial is R = (1 + (2Ay + B)^2)^(3/2) / |2A|.

The next important step for this algorithm is to convert from the image domain (in pixels) to the real world domain (in meters). In order to calculate this conversion factor, I used the equivalency that the image width of the lane (700 pixels) is the standard US lane width of 3.7 m and the image height (720 pixels) is approximately 30 m.

With the unit conversion ratios and the equation for radius of curvature, it was possible to calculate the radius in real world units (m).

Finally, in order to calculate the vehicle offset it simply required an assumption that the center of the frame (camera) was aligned with the center of the vehicle. Therefore, measuring the number of pixels between the center of the frame and the left lane line (and converting to real world units) would yield the lane center offset. A sample of the output of these calculations may be seen in the next section.

#### 7. Temporal Filtering and Final Result

Finally, in order implement this pipeline robustly, I decided to include some temporal filtering and additional search logic. Specifically...

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
