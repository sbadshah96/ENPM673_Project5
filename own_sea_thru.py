import collections
import cv2 as cv2
import sys
import argparse
import numpy as np
import sklearn as sk
import scipy as sp
import scipy.optimize
import scipy.stats
import math
from PIL import Image
# import rawpy
import matplotlib
from matplotlib import pyplot as plt
from skimage import exposure
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle, estimate_sigma
from skimage.morphology import closing, opening, erosion, dilation, disk, diamond, square

def readImage(image, depth_image, size_limit = 1024):
    image = cv2.imread(image)
    depth_image = cv2.imread(depth_image)
    resized_image = cv2.resize(image,(size_limit,size_limit),interpolation=cv2.INTER_AREA)
    resized_depth_image = cv2.resize(depth_image,(size_limit,size_limit),interpolation=cv2.INTER_AREA)
    return np.float32(resized_image)/255.0, resized_depth_image

def depth_preprocess(depth_image, min_depth, max_depth):
    z_min = np.min(depth_image) + (min_depth * (np.max(depth_image) - np.min(depth_image)))
    z_max = np.min(depth_image) + (max_depth * (np.max(depth_image) - np.min(depth_image)))
    if max_depth != 0:
        depth_image[depth_image == 0] = z_max
    depth_image[depth_image < z_min] = 0
    return depth_image

def backscatter_estimation(image,depth_image,fraction=0.01,max_points=30,min_depth_perc=0.0,limit=10):
    z_max, z_min = np.max(depth_image), np.min(depth_image)
    min_depth = z_min + (min_depth_perc * (z_max - z_min))
    z_ranges = np.linspace(z_min,z_max,limit + 1)
    image_norms = np.mean(image, axis=2)
    R_point = []
    G_point = []
    B_point = []
    for i in range(len(z_ranges) - 1):
        a, b = z_ranges[i], z_ranges[i+1]
        indices = np.asarray(np.logical_and(depth_image > min_depth, np.logical_and(depth_image >= a,depth_image <= b ))).nonzero()
        indiced_norm, indiced_image, indiced_depth_image = image_norms[indices], image[indices], depth_image[indices]
        combined_data = sorted(zip(indiced_norm, indiced_image, indiced_depth_image), key=lambda x: x[0])
        points = combined_data[:min(math.ceil(fraction * len(combined_data)), max_points)]
        for _, image_point, depth_image_point in points:
            R_point.extend((depth_image_point, image_point[0]))
            G_point.extend((depth_image_point, image_point[1]))
            B_point.extend((depth_image_point, image_point[2]))
    return np.array(R_point), np.array(G_point), np.array(B_point)


def find_backscatter_values(BS_pts, depth_image, iterations=10, max_mean_loss_fraction=0.1):
    BS_depth_val, BS_img_val = BS_pts[:, 0], BS_pts[:, 1]
    z_max, z_min = np.max(depth_image), np.min(depth_image)
    max_mean_loss = max_mean_loss_fraction * (z_max - z_min)
    coeffs = None
    best_loss = np.inf
    lower_limit = [0,0,0,0]
    upper_limit = [1,5,1,5]

    def estimate_coeffs(depth_img_val, B_inf, beta_B, J_c, beta_D):
        img_val = (B_inf * (1 - np.exp(-1 * beta_B * depth_img_val))) + (J_c * np.exp(-1 * beta_D * depth_img_val))
        return img_val
    def estimate_loss(B_inf, beta_B, J_c, beta_D):
        loss_val = np.mean(np.abs(BS_img_val - estimate_coeffs(BS_depth_val, B_inf, beta_B, J_c, beta_D)))
        return loss_val
    
    for _ in range(iterations):
        try:
            est_coeffs, _ = sp.optimize.curve_fit(
                f=estimate_coeffs,
                xdata=BS_depth_val,
                ydata=BS_img_val,
                p0=np.random.random(4) * upper_limit,
                bounds=(lower_limit, upper_limit),
            )
            current_loss = estimate_loss(*est_coeffs)
            if current_loss < best_loss:
                best_loss = current_loss
                coeffs = est_coeffs
        except RuntimeError as re:
            print(re, file=sys.stderr)

    if best_loss > max_mean_loss:
        print('Warning: could not find accurate reconstruction. Switching to linear model.', flush=True)
        m, b, _, _, _ = sp.stats.linregress(BS_depth_val, BS_img_val)
        y = (m * depth_image) + b
        return y, np.array([m, b])
    return estimate_coeffs(depth_image, *coeffs), coeffs


min_depth = 0.1
max_depth = 1

if __name__ == "__main__":
    image = "4_img_.png"
    depth_image = "4_img_.png"
    
    image, depth_image = readImage(image,depth_image)

    depth_image = depth_preprocess(depth_image,min_depth,max_depth)

    R_back,B_back,G_back = backscatter_estimation(image,depth_image,fraction=0.01,min_depth_perc=min_depth)

    BS_red, red_coeffs = find_backscatter_values(R_back, depth_image, iterations=25)
    BS_green, green_coeffs = find_backscatter_values(G_back, depth_image, iterations=25)
    BS_blue, blue_coeffs = find_backscatter_values(B_back, depth_image, iterations=25)



