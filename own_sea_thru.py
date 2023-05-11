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


def calculate_backscatter_values(BS_pts, depth_image, iterations=10, max_mean_loss_fraction=0.1):
    BS_depth_val, BS_img_val = BS_pts[:, 0], BS_pts[:, 1]
    z_max, z_min = np.max(depth_image), np.min(depth_image)
    max_mean_loss = max_mean_loss_fraction * (z_max - z_min)
    coeffs = None
    min_loss = np.inf
    lower_limit = [0,0,0,0]
    upper_limit = [1,5,1,5]

    def estimate_coeffs(depth_img_val, B_inf, beta_B, J_c, beta_D):
        img_val = (B_inf * (1 - np.exp(-1 * beta_B * depth_img_val))) + (J_c * np.exp(-1 * beta_D * depth_img_val))
        return img_val
    def estimate_loss(B_inf, beta_B, J_c, beta_D):
        loss_val = np.mean(np.abs(BS_img_val - estimate_coeffs(BS_depth_val, B_inf, beta_B, J_c, beta_D)))
        return loss_val
    
    for i in range(iterations):
        try:
            est_coeffs, _ = sp.optimize.curve_fit(
                f=estimate_coeffs,
                xdata=BS_depth_val,
                ydata=BS_img_val,
                p0=np.random.random(4) * upper_limit,
                bounds=(lower_limit, upper_limit),
            )
            current_loss = estimate_loss(*est_coeffs)
            if current_loss < min_loss:
                min_loss = current_loss
                coeffs = est_coeffs
        except RuntimeError as re:
            print(re, file=sys.stderr)

    if min_loss > max_mean_loss:
        print('Warning: could not find accurate reconstruction. Switching to linear model.', flush=True)
        m, b, _, _, _ = sp.stats.linregress(BS_depth_val, BS_img_val)
        y = (m * depth_image) + b
        return y, np.array([m, b])
    return estimate_coeffs(depth_image, *coeffs), coeffs

def find_closest_label(new_map, begin_x, begin_y):
    t = collections.deque()
    t.append((begin_x, begin_y))
    n_map_mask = np.zeros_like(new_map).astype(np.bool)
    while not len(t) == 0:
        x, y = t.pop()
        if 0 <= x < new_map.shape[0] and 0 <= y < new_map.shape[1]:
            if new_map[x, y] != 0:
                return new_map[x, y]
            n_map_mask[x, y] = True
            if 0 <= x < new_map.shape[0] - 1:
                x2, y2 = x + 1, y
                if not n_map_mask[x2, y2]:
                    t.append((x2, y2))
            if 1 <= x < new_map.shape[0]:
                x2, y2 = x - 1, y
                if not n_map_mask[x2, y2]:
                    t.append((x2, y2))
            if 0 <= y < new_map.shape[1] - 1:
                x2, y2 = x, y + 1
                if not n_map_mask[x2, y2]:
                    t.append((x2, y2))
            if 1 <= y < new_map.shape[1]:
                x2, y2 = x, y - 1
                if not n_map_mask[x2, y2]:
                    t.append((x2, y2))

def construct_neighbor_map(depth_image, epsilon=0.05):
    eps = (np.max(depth_image) - np.min(depth_image)) * epsilon
    #Create a new array of depth image shape with zeroes 
    new_map = np.zeros_like(depth_image).astype(np.int32)
    neighborhoods = 1
    #Run the loop until new map has zero value
    while np.any(new_map == 0):
        x_locs, y_locs = np.where(new_map == 0)
        start_index = np.random.randint(0, len(x_locs))
        start_x, start_y = x_locs[start_index], y_locs[start_index]
        que = collections.deque()
        que.append((start_x, start_y))
        while bool(que) == True:
            x, y = que.pop()
            if np.abs(depth_image[x, y] - depth_image[start_x, start_y]) <= eps:
                new_map[x, y] = neighborhoods
                if 0 <= x < depth_image.shape[0] - 1:
                    x2, y2 = x + 1, y
                    if new_map[x2, y2] == 0:
                        que.append((x2, y2))
                if 1 <= x < depth_image.shape[0]:
                    x2, y2 = x - 1, y
                    if new_map[x2, y2] == 0:
                        que.append((x2, y2))
                if 0 <= y < depth_image.shape[1] - 1:
                    x2, y2 = x, y + 1
                    if new_map[x2, y2] == 0:
                        que.append((x2, y2))
                if 1 <= y < depth_image.shape[1]:
                    x2, y2 = x, y - 1
                    if new_map[x2, y2] == 0:
                        que.append((x2, y2))
        neighborhoods += 1
    zeros_arr = sorted(zip(*np.unique(new_map[depth_image == 0], return_counts=True)), key=lambda x: x[1], reverse=True)
    if len(zeros_arr) > 0:
        new_map[new_map == zeros_arr[0][0]] = 0 
    return new_map, neighborhoods - 1

def refining_neighbor_map(new_map, min_size=10, radius=3):
    pts, counts = np.unique(new_map, return_counts=True)
    neighbor_sizes = sorted([(pt, count) for pt, count in zip(pts, counts)], key=lambda x: x[1], reverse=True)
    refined_new_map = np.zeros_like(new_map)
    total_labels = 1
    for label, size in neighbor_sizes:
        if size >= min_size and label != 0:
            refined_new_map[new_map == label] = total_labels
            total_labels += 1
    for label, size in neighbor_sizes:
        if size < min_size and label != 0:
            for x, y in zip(*np.where(new_map == label)):
                refined_new_map[x, y] = find_closest_label(refined_new_map, x, y)
    refined_n = closing(refined_new_map, square(radius))
    return refined_n, total_labels - 1

def estimate_illumination_map(image, BS_val, neighbor_map, num_neighbor, p=0.5, f=2.0, max_iters=100, tol=1E-5):
    direct_sig = image - BS_val
    avg_space_clr = np.zeros(image.shape) # Test Later
    avg_space_clr_prime = np.copy(avg_space_clr)
    sizes = np.zeros(num_neighbor)
    indices = [None] * num_neighbor
    for num in range(1, num_neighbor + 1):
        indices[num - 1] = np.where(neighbor_map == num)   # Test with this also np.asarray(a == label)
        sizes[num - 1] = np.size(indices[num - 1][0])
    for i in range(max_iters):
        for num in range(1, num_neighbor + 1):
            index = indices[num - 1]
            size = sizes[num - 1] - 1
            avg_space_clr_prime[index] = (1 / size) * (np.sum(avg_space_clr[index]) - avg_space_clr[index])
        new_avg_space_clr = (direct_sig * p) + (avg_space_clr_prime * (1 - p))
        if(np.max(np.abs(avg_space_clr - new_avg_space_clr)) < tol):
            break
        avg_space_clr = new_avg_space_clr
    return f * denoise_bilateral(np.maximum(0, avg_space_clr))

def data_filter(X, Y, radius_frac=0.01):
    indexes = np.argsort(X)
    X_s = X[indexes]
    Y_s = Y[indexes]
    x_maximum, x_minimum = np.max(X), np.min(X)
    radius = (radius_frac * (x_maximum - x_minimum))
    dS = np.cumsum(X_s - np.roll(X_s, (1,)))
    dX = [X_s[0]]
    dY = [Y_s[0]]
    temp_X = []
    temp_Y = []
    position = 0
    for i in range(1, dS.shape[0]):
        if dS[i] - dS[position] >= radius:
            temp_X.append(X_s[i])
            temp_Y.append(Y_s[i])
            indexes = np.argsort(temp_Y)
            med_idx = len(indexes) // 2
            dX.append(temp_X[med_idx])
            dY.append(temp_Y[med_idx])
            position = i
        else:
            temp_X.append(X_s[i])
            temp_Y.append(Y_s[i])
    return np.array(dX), np.array(dY)


min_depth = 0.1
max_depth = 1

if __name__ == "__main__":
    image = "4_img_.png"
    depth_image = "4_img_.png"
    
    image, depth_image = readImage(image,depth_image)

    depth_image = depth_preprocess(depth_image,min_depth,max_depth)

    R_back,B_back,G_back = backscatter_estimation(image,depth_image,fraction=0.01,min_depth_perc=min_depth)

    BS_red, red_coeffs = calculate_backscatter_values(R_back, depth_image, iterations=25)
    BS_green, green_coeffs = calculate_backscatter_values(G_back, depth_image, iterations=25)
    BS_blue, blue_coeffs = calculate_backscatter_values(B_back, depth_image, iterations=25)

    new_map, _ = construct_neighbor_map(depth_image, 0.08)

    new_map, n = refining_neighbor_map(new_map, 50)

    Red_illumination = estimate_illumination_map(image[:, :, 0], BS_red, new_map, n, p=0.5, f=2.0, max_iters=100, tol=1E-5)
    Green_illumination = estimate_illumination_map(image[:, :, 1], BS_green, new_map, n, p=0.5, f=2.0, max_iters=100, tol=1E-5)
    Blue_illumination = estimate_illumination_map(image[:, :, 2], BS_blue, new_map, n, p=0.5, f=2.0, max_iters=100, tol=1E-5)
    
    Total_illumination = np.stack([Red_illumination, Green_illumination, Blue_illumination], axis=2)


