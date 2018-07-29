import cv2

def warper(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped
	
	
	
	
	# 1. Camera Calibration
		# a. Compute camera calibration matrix and distortion coefficients
	# 2. Distortion Correction
		# a. Use coefficients to undistort each frame
	# 3. Color & Gradient Threshold
		# a. Apply combination of color and gradient threshold to generate a binary image that make lane lines visable
	# 4. Perspective Transform
		# a. Select image with straight lines
		# b. Select four (source) points that make up a trapezoid which will be transformed into a rectangle from the top-down view
		# c. Use the same four source points to tranform every plane (lane lines should always be parallel, even when curved)
	# 5. Identify lane lines
		# a. Calculate histogram of binary points in bottom half of image to identify starting points of lane linesimport numpy as np
			import matplotlib.image as mpimg
			import matplotlib.pyplot as plt

			# Load our image
			# `mpimg.imread` will load .jpg as 0-255, so normalize back to 0-1
			img = mpimg.imread('warped_example.jpg')/255

			def hist(img):
				# Grab only the bottom half of the image
				# Lane lines are likely to be mostly vertical nearest to the car
				bottom_half = img[img.shape[0]//2:,:]

				# Sum across image pixels vertically - make sure to set an `axis`
				# i.e. the highest areas of vertical lines should be larger values
				histogram = np.sum(bottom_half, axis=0)
				
				return histogram

			# Create histogram of image binary activations
			histogram = hist(img)

			# Visualize the resulting histogram
			plt.plot(histogram)
		# b. Create sliding windows acorss each lane to determine lane position going away (forward) from the vehicle
			import numpy as np
			import matplotlib.image as mpimg
			import matplotlib.pyplot as plt
			import cv2

			# Load our image
			binary_warped = mpimg.imread('warped_example.jpg')

			def find_lane_pixels(binary_warped):
				# Take a histogram of the bottom half of the image
				histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
				# Create an output image to draw on and visualize the result
				out_img = np.dstack((binary_warped, binary_warped, binary_warped))
				# Find the peak of the left and right halves of the histogram
				# These will be the starting point for the left and right lines
				midpoint = np.int(histogram.shape[0]//2)
				leftx_base = np.argmax(histogram[:midpoint])
				rightx_base = np.argmax(histogram[midpoint:]) + midpoint

				# HYPERPARAMETERS
				# Choose the number of sliding windows
				nwindows = 9
				# Set the width of the windows +/- margin
				margin = 100
				# Set minimum number of pixels found to recenter window
				minpix = 50

				# Set height of windows - based on nwindows above and image shape
				window_height = np.int(binary_warped.shape[0]//nwindows)
				# Identify the x and y positions of all nonzero pixels in the image
				nonzero = binary_warped.nonzero()
				nonzeroy = np.array(nonzero[0])
				nonzerox = np.array(nonzero[1])
				# Current positions to be updated later for each window in nwindows
				leftx_current = leftx_base
				rightx_current = rightx_base

				# Create empty lists to receive left and right lane pixel indices
				left_lane_inds = []
				right_lane_inds = []

				# Step through the windows one by one
				for window in range(nwindows):
					# Identify window boundaries in x and y (and right and left)
					win_y_low = binary_warped.shape[0] - (window+1)*window_height
					win_y_high = binary_warped.shape[0] - window*window_height
					win_xleft_low = leftx_current - margin
					win_xleft_high = leftx_current + margin
					win_xright_low = rightx_current - margin
					win_xright_high = rightx_current + margin
					
					# Draw the windows on the visualization image
					cv2.rectangle(out_img,(win_xleft_low,win_y_low),
					(win_xleft_high,win_y_high),(0,255,0), 2) 
					cv2.rectangle(out_img,(win_xright_low,win_y_low),
					(win_xright_high,win_y_high),(0,255,0), 2) 
					
					# Identify the nonzero pixels in x and y within the window #
					good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
					(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
					good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
					(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
					
					# Append these indices to the lists
					left_lane_inds.append(good_left_inds)
					right_lane_inds.append(good_right_inds)
					
					# If you found > minpix pixels, recenter next window on their mean position
					if len(good_left_inds) > minpix:
						leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
					if len(good_right_inds) > minpix:        
						rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

				# Concatenate the arrays of indices (previously was a list of lists of pixels)
				try:
					left_lane_inds = np.concatenate(left_lane_inds)
					right_lane_inds = np.concatenate(right_lane_inds)
				except ValueError:
					# Avoids an error if the above is not implemented fully
					pass

				# Extract left and right line pixel positions
				leftx = nonzerox[left_lane_inds]
				lefty = nonzeroy[left_lane_inds] 
				rightx = nonzerox[right_lane_inds]
				righty = nonzeroy[right_lane_inds]

				return leftx, lefty, rightx, righty, out_img


			def fit_polynomial(binary_warped):
				# Find our lane pixels first
				leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

				# Fit a second order polynomial to each using `np.polyfit`
				left_fit = np.polyfit(lefty, leftx, 2)
				right_fit = np.polyfit(righty, rightx, 2)

				# Generate x and y values for plotting
				ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
				try:
					left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
					right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
				except TypeError:
					# Avoids an error if `left` and `right_fit` are still none or incorrect
					print('The function failed to fit a line!')
					left_fitx = 1*ploty**2 + 1*ploty
					right_fitx = 1*ploty**2 + 1*ploty

				## Visualization ##
				# Colors in the left and right lane regions
				out_img[lefty, leftx] = [255, 0, 0]
				out_img[righty, rightx] = [0, 0, 255]

				# Plots the left and right polynomials on the lane lines
				plt.plot(left_fitx, ploty, color='yellow')
				plt.plot(right_fitx, ploty, color='yellow')

				return out_img


			out_img = fit_polynomial(binary_warped)

			plt.imshow(out_img)
		# c. Step through subsequent frames and use the first polynomial to perform a search for the new ones
			import cv2
			import numpy as np
			import matplotlib.image as mpimg
			import matplotlib.pyplot as plt

			# Load our image - this should be a new frame since last time!
			binary_warped = mpimg.imread('warped_example.jpg')

			# Polynomial fit values from the previous frame
			# Make sure to grab the actual values from the previous step in your project!
			left_fit = np.array([ 2.13935315e-04, -3.77507980e-01,  4.76902175e+02])
			right_fit = np.array([4.17622148e-04, -4.93848953e-01,  1.11806170e+03])

			def fit_poly(img_shape, leftx, lefty, rightx, righty):
				 ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
				left_fit = np.polyfit(lefty, leftx, 2)
				right_fit = np.polyfit(righty, rightx, 2)
				# Generate x and y values for plotting
				ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
				### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
				left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
				right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
				
				return left_fitx, right_fitx, ploty

			def search_around_poly(binary_warped):
				# HYPERPARAMETER
				# Choose the width of the margin around the previous polynomial to search
				# The quiz grader expects 100 here, but feel free to tune on your own!
				margin = 100

				# Grab activated pixels
				nonzero = binary_warped.nonzero()
				nonzeroy = np.array(nonzero[0])
				nonzerox = np.array(nonzero[1])
				
				### TO-DO: Set the area of search based on activated x-values ###
				### within the +/- margin of our polynomial function ###
				### Hint: consider the window areas for the similarly named variables ###
				### in the previous quiz, but change the windows to our new search area ###
				left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
								left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
								left_fit[1]*nonzeroy + left_fit[2] + margin)))
				right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
								right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
								right_fit[1]*nonzeroy + right_fit[2] + margin)))
				
				# Again, extract left and right line pixel positions
				leftx = nonzerox[left_lane_inds]
				lefty = nonzeroy[left_lane_inds] 
				rightx = nonzerox[right_lane_inds]
				righty = nonzeroy[right_lane_inds]

				# Fit new polynomials
				left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
				
				## Visualization ##
				# Create an image to draw on and an image to show the selection window
				out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
				window_img = np.zeros_like(out_img)
				# Color in left and right line pixels
				out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
				out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

				# Generate a polygon to illustrate the search window area
				# And recast the x and y points into usable format for cv2.fillPoly()
				left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
				left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
										  ploty])))])
				left_line_pts = np.hstack((left_line_window1, left_line_window2))
				right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
				right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
										  ploty])))])
				right_line_pts = np.hstack((right_line_window1, right_line_window2))

				# Draw the lane onto the warped blank image
				cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
				cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
				result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
				
				# Plot the polynomial lines onto the image
				plt.plot(left_fitx, ploty, color='yellow')
				plt.plot(right_fitx, ploty, color='yellow')
				## End visualization steps ##
				
				return result

			# Run image through the pipeline
			# Note that in your project, you'll also want to feed in the previous fits
			result = search_around_poly(binary_warped)

			# View your output
			plt.imshow(result)
			
			# One thing to consider in our current implementation of sliding window search is what happens when we arrive at the left or right edge of an image, such as when there is a large curve on the road ahead. 
			# If minpix is not achieved (i.e. the curve ran off the image), the starting position of our next window doesn't change, so it is just positioned directly above the previous window. 
			# This will repeat for however many windows are left in nwindows, stacking the sliding windows vertically against the side of the image, and likely leading to an imperfect polynomial fit.
			# Can you think of a way to solve this issue? If you want to tackle the curves on the harder challenge video as part of the project, you might want to include this in your lane finding algorithm.
	