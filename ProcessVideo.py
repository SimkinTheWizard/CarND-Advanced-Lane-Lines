import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
#%matplotlib qt
from matplotlib import gridspec
from joblib import Parallel, delayed
import multiprocessing
import gc

gc.collect()

video_file="project_video.mp4"
output_file="project_video_out1.avi"

# Global definitions
ym_per_pix = 40 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

window_width = 50
window_height = 150  # Break image into 9 vertical layers since image height is 720
margin = 100  # How much to slide left and right for searching

sobel_thresh=20
s_thresh = 140
l_thresh = 190

frame_shape=(1280,720)
CPU_BATCH_SIZE=16


verbose = False
# --------------------------------------
# a utility function for plotting opencv images with matplotlib
def mp(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


def correct_perspective(img, M, img_size):
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    warped = cv2.warpPerspective(undistorted, M, img_size, flags=cv2.INTER_LINEAR)
    return warped


def set_up_perspective_transform():
    #gray = cv2.cvtColor(cv2.imread(test_images[0]),cv2.COLOR_BGR2GRAY)
    img_size = frame_shape # (gray.shape[1], gray.shape[0])
    #print (img_size)
    #src=np.float32([(450,450),(1000,450),(2000,700),(200,700)])
    #dst=np.float32([(0,0),(1200,0),(1200,700),(00,700)])
    offset = 5
    dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
                                         [img_size[0]-offset, img_size[1]-offset],
                                         [offset, img_size[1]-offset]])
    #src=np.float32([(590,440),(700,440),(1300,640),(90,640)])
    y_min = 445
    y_max = 640
    src=np.float32([(575,y_min),(713,y_min),(1300,y_max),(90,y_max)])
    M = cv2.getPerspectiveTransform(src, dst)
    return M,img_size

def sobel_edges(img, thresh_min = 20, thresh_max = 100):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sxbinary

def s_magnitude(img, s_thresh_min = 150, s_thresh_max = 255):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    return s_binary

def l_magnitude(img, l_thresh_min = 190, l_thresh_max = 255):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:,:,1]
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1
    return l_binary

def combine(sxbinary, s_binary):
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

def combine3(sxbinary, s_binary,h_binary):
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1) | (h_binary == 1)] = 1
    return combined_binary

def denoise(combined):
    kernel = np.ones((3,3),np.uint8)
    combined_open = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    return combined_open

def binarize(img):
    s_mag = s_magnitude(img,s_thresh)
    sobel = sobel_edges(img,sobel_thresh)
    l_mag = l_magnitude(img,l_thresh)
    #combined = combine(sobel,s_mag)
    combined = combine3(sobel,s_mag,l_mag)
    combined_open = denoise(combined)
    return combined_open, s_mag, sobel, l_mag

def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output


def find_window_centroids(image, window_width=50, window_height=100, margin=100):
    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(1 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    r_sum = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(image.shape[1] / 2)

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(image.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            image[int(image.shape[0] - (level + 1) * window_height):int(image.shape[0] - level * window_height), :],
            axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        # Add what we found for that layer
        window_centroids.append((l_center, r_center))

    return window_centroids


def find_lane_pixels(binary, l_points, r_points):
    l_pixels = np.zeros_like(l_points)
    r_pixels = np.zeros_like(r_points)

    l_pixels[(binary > 0) & (l_points > 0)] = 255
    r_pixels[(binary > 0) & (r_points > 0)] = 255
    return l_pixels, r_pixels





def get_pixel_locations(lane_pixels):
    loc_y = []
    loc_x = []
    # print (lane_pixels.shape)
    for y in range(lane_pixels.shape[0]):
        for x in range(lane_pixels.shape[1]):
            if (lane_pixels[y][x] > 0):
                loc_y.append(y)
                loc_x.append(x)
    return np.asarray(loc_y), np.asarray(loc_x)


def fit_curve_to_locations(lane_pixels):
    loc_y, loc_x = get_pixel_locations(lane_pixels)
    fit = np.polyfit(loc_y, loc_x, 2)
    # print(fit)
    return fit, loc_y, loc_x


def fit_curve_to_real_locations(lane_pixels):
    loc_y, loc_x = get_pixel_locations(lane_pixels)
    fit = np.polyfit(loc_y * ym_per_pix, loc_x * xm_per_pix, 2)
    # print(fit)
    return fit


def curve_fit_to_image(poly_fit, binary_image_shape):
    frame = np.zeros_like(binary_image_shape)
    # print(frame.shape)
    # print (poly_fit[0])
    for y in range(frame.shape[0] - 1):
        x = poly_fit[0] * y ** 2 + poly_fit[1] * y + poly_fit[2]
        frame[y,max(0,(min(int(x),frame.shape[1]-1)))] = 255
        frame[y,max(0,min(int(x+1),frame.shape[1]-1))] = 255
        frame[y,max(0,min(int(x-1),frame.shape[1]-1))] = 255
    return frame


def curve_fit_to_array(poly_fit, binary_image_shape):
    arr_x = []
    for y in range(binary_image_shape.shape[0] - 1):
        x = poly_fit[0] * y ** 2 + poly_fit[1] * y + poly_fit[2]
        arr_x.append(min(x, binary_image_shape.shape[1] - 1))

    return range(binary_image_shape.shape[0] - 1), np.asarray(arr_x)


def calculate_curvature_radius(l_fit, r_fit):
    # Define conversions in x and y from pixels space to meters

    # Fit new polynomials to x,y in world space
    # left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    # right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    y_eval = 720
    # y_eval = np.max(ploty)
    left_curverad = ((1 + (2 * l_fit[0] * y_eval * ym_per_pix + l_fit[1]) ** 2) ** 1.5) / np.absolute(2 * l_fit[0])
    right_curverad = ((1 + (2 * r_fit[0] * y_eval * ym_per_pix + r_fit[1]) ** 2) ** 1.5) / np.absolute(2 * r_fit[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
    return left_curverad, right_curverad

def calculate_relative_position(window_centroids):
    position=  1280/2 - ((window_centroids[0][1] - window_centroids[0][0])/2 + window_centroids[0][0])
    position_real_world = position * xm_per_pix
    return position_real_world

def calculate_weighted_average(left_curverad, right_curverad, l_locations, r_locations):
    return (left_curverad*l_locations.size + right_curverad * r_locations.size) / (l_locations.size + r_locations.size)


def set_inverse_transform():
    #gray = cv2.cvtColor(cv2.imread(test_images[0]),cv2.COLOR_BGR2GRAY)
    img_size = frame_shape # (gray.shape[1], gray.shape[0])
    #print (img_size)
    #src=np.float32([(450,450),(1000,450),(2000,700),(200,700)])
    #dst=np.float32([(0,0),(1200,0),(1200,700),(00,700)])
    offset = 5
    dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
                                         [img_size[0]-offset, img_size[1]-offset],
                                         [offset, img_size[1]-offset]])
    #src=np.float32([(590,440),(700,440),(1300,640),(90,640)])
    y_min = 445
    y_max = 640
    src=np.float32([(575,y_min),(713,y_min),(1300,y_max),(90,y_max)])
    Minv = cv2.getPerspectiveTransform(dst,src )
    return Minv,img_size


def draw_lane_polygon(frame, warped, Minv, l_x,l_y, r_x,r_y ):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([l_x, l_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([r_x, r_y])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (frame.shape[1], frame.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(frame, 1, newwarp, 0.3, 0)
    #plt.imshow(result)
    return result

def process_frame(img):
    warped = correct_perspective(img, M, img_size)
    binary, s_mag, sobel, l_mag = binarize(warped)


    window_centroids = find_window_centroids(binary, window_width,window_height,margin)
    if verbose:
        output2 = np.zeros((1280, int((720*4)/3), 3))

    output = np.zeros((1280, 720, 3))
    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(binary)
        r_points = np.zeros_like(binary)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, binary, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, binary, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        # Draw the results
        l_pixels, r_pixels = find_lane_pixels(binary, l_points, r_points)
        l_poly, l_loc_y, l_loc_x = fit_curve_to_locations(l_pixels)
        r_poly, r_loc_y, r_loc_x = fit_curve_to_locations(r_pixels)

        l_fit = curve_fit_to_image(l_poly, binary)
        r_fit = curve_fit_to_image(r_poly, binary)

        l_y, l_x = curve_fit_to_array(l_poly, binary)
        r_y, r_x = curve_fit_to_array(r_poly, binary)

        l_poly_r = fit_curve_to_real_locations(l_pixels)
        r_poly_r = fit_curve_to_real_locations(r_pixels)
        left_curverad, right_curverad = calculate_curvature_radius(l_poly_r, r_poly_r)

        curvature_radius = calculate_weighted_average(left_curverad, right_curverad, l_loc_x, r_loc_x)
        # print(curvature_radius)
        relative_position = calculate_relative_position(window_centroids)
        #print (relative_position)

        template1 = np.array(r_fit, np.uint8)  # add both left and right window pixels together
        template2 = np.array(l_fit, np.uint8)  # add both left and right window pixels together
        zero_channel = np.zeros_like(template1)  # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template1, template2)), np.uint8)  # make window pixels green
        warpage = np.dstack((binary, binary, binary)) * 255  # making the original road pixels 3 color channels

        # output = cv2.addWeighted(warpage, 0.5, template, 0.99, 0.0) # overlay the orignal road image with window results
        # output = np.zeros_like(img)
        output = draw_lane_polygon(img, binary, Minv, l_x, l_y, r_x, r_y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        cv2.putText(output, ("Radius of curvature "+str(curvature_radius) + " m"), (100, 50), font, font_scale, (255, 255, 255),
                    2, cv2.LINE_AA)
        cv2.putText(output, ("Relative position  "+str(relative_position) + " m"), (100, 100), font, font_scale, (255, 255, 255),
                    2, cv2.LINE_AA)

        if verbose:
	    
            #warped_small = cv2.resize(warped, (320,180), interpolation=cv2.INTER_NEAREST)
            binary_small = cv2.resize(binary, (320,180), interpolation=cv2.INTER_NEAREST)*255
            #zeros_small = output = np.zeros_like(binary_small)
            s_mag_small = cv2.resize(s_mag, (320,180), interpolation=cv2.INTER_NEAREST)*255
            l_mag_small = cv2.resize(l_mag, (320,180), interpolation=cv2.INTER_NEAREST)*255
            sobel_small = cv2.resize(sobel, (320,180), interpolation=cv2.INTER_NEAREST)*255

            verbose_small = np.concatenate((binary_small, s_mag_small, l_mag_small, sobel_small), axis=1)


            output2gray = np.array(cv2.merge((verbose_small, verbose_small, verbose_small)), np.uint8)
            output2 = np.concatenate((output2gray,output),axis=0)


        # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((binary,binary,binary)),np.uint8)
    if verbose:
        return output2
    return output

# --------------------------------------


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')

print("Reading camera images and finding chessboard corners")
# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        #cv2.imshow('img',img)
        #cv2.waitKey(500)
        #plt.figure()
        #plt.imshow(img, cmap='gray')
        #plt.show()

print("Calibrating camera")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


print("Setting up perspective transforms")
M, img_size = set_up_perspective_transform()
Minv, dummy = set_inverse_transform()


print("Opening video file for processing")
capture = cv2.VideoCapture(video_file)
length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
print( "The video has " + str(length) + " frames" )



# Define the codec and create VideoWriter object

fps = 30

if verbose:
    capSize = (1280, 900)
else:
    capSize = (1280, 720)
fourcc = cv2.VideoWriter_fourcc('x', '2', '6', '4')
writer = cv2.VideoWriter()
success = writer.open(output_file,fourcc,fps,capSize,True)
if not success:
    print("Could not open output video file for writing")
    exit(-1)




frame_count = 0
num_cores = multiprocessing.cpu_count()
print ("This cpu has "+str(num_cores)+" cores")


batch_frames = []
while(capture.isOpened()):
    ret, img = capture.read()
    if ret==True:
        frame_count = frame_count + 1
        print("Frame " + str(frame_count) + " of " + str(length) + "\r")

        #if frame_count < 1030:
        #    continue
        #batch_frames.append(img)
        output = process_frame(img)
        writer.write(output)
        cv2.imshow('frame', output)
        #if (len(batch_frames) >= CPU_BATCH_SIZE):
        #    results = Parallel(n_jobs=num_cores)(delayed(process_frame)(batch_frames[i]) for i in range(len(batch_frames)))
        #    results = Parallel(                delayed(process_frame)(batch_frames[i]) for i in range(len(batch_frames)))
        #    print ("Processed " + str(len(results)) + "frames")
        #    for i in range( len(batch_frames)):
        #        output = np.asarray(results[i])
        #        writer.write(output)
        #        cv2.imshow('frame',results[i])
        #        cv2.waitKey(1)
        #    batch_frames = []
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

print()
print("Finish")

# Release everything if job is finished
capture.release()
writer.release()
