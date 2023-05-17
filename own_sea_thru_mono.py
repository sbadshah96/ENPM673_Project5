import collections
import cv2 as cv2
import sys
import numpy as np
import scipy as sp
import math
import PIL.Image as pil
from matplotlib import pyplot as plt
from skimage import exposure
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle, estimate_sigma
from skimage.morphology import closing, disk, square
import tensorflow as tf
import torch
from PIL import Image
from torchvision import transforms, datasets

sys.path.append('/Users/shreejay/Desktop/UMD/ENPM673/Projects/Project5/code/implementation/ENPM673_Project5/deps/monodepth2')
import deps.monodepth2.networks as networks
from deps.monodepth2.layers import disp_to_depth
from deps.monodepth2.utils import download_model_if_doesnt_exist

import warnings
warnings.simplefilter("ignore",category=UserWarning)

def readImage(image, size_limit = 256):
    image = cv2.imread(image)
    resized_image = cv2.resize(image,(size_limit,size_limit),interpolation=cv2.INTER_AREA)
    resized_image = cv2.cvtColor(resized_image,cv2.COLOR_BGR2RGB)
    resized_image = tf.image.convert_image_dtype(resized_image,tf.float32)

    device = torch.device("cpu")

    # new_model = tf.keras.models.load_model('/Users/shreejay/Desktop/UMD/ENPM673/Projects/Project5/code/my_model')
    
    model_path = "/Users/shreejay/Desktop/UMD/ENPM673/Projects/Project5/code/sea-thru/models"
    encoder_path = "/Users/shreejay/Desktop/UMD/ENPM673/Projects/Project5/code/sea-thru/models/mono_1024x320/encoder.pth"
    depth_decoder_path = "/Users/shreejay/Desktop/UMD/ENPM673/Projects/Project5/code/sea-thru/models/mono_1024x320/depth.pth"

    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()
    input_image = np.array(resized_image)

    # Load image and preprocess
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    print('Preprocessed image', flush=True)
    print(input_image.shape)

    # PREDICTION
    input_image = input_image.to(device)
    features = encoder(input_image)
    outputs = depth_decoder(features)
    depth_image = outputs[("disp", 0)]
    depth_image = depth_image.squeeze().cpu().detach().numpy()
    
    cmap = plt.cm.jet
    cmap.set_bad(color="black")
    d_image = data_generation(resized_image)
    return np.float32(resized_image)/255.0, np.array(depth_image)

def data_generation(image):
        x = np.empty((1, 256, 256, 3))
        x[0,] = image
        return x

def depth_preprocess(depth_image, min_depth, max_depth):
    # print("depth zero zero: ", depth_image.shape)
    z_min = np.min(depth_image) + (min_depth * (np.max(depth_image) - np.min(depth_image)))
    z_max = np.min(depth_image) + (max_depth * (np.max(depth_image) - np.min(depth_image)))
    if max_depth != 0:
        depth_image[depth_image == 0] = z_max
    depth_image[depth_image < z_min] = 0
    return depth_image

def backscatter_estimation(image,depth_image,fraction=0.01,max_points=30,min_depth_perc=0.01,limit=10):
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
            R_point.append((depth_image_point, image_point[0]))
            G_point.append((depth_image_point, image_point[1]))
            B_point.append((depth_image_point, image_point[2]))
        # R_point.extend([(z, p[0]) for n, p, z in points])
        # G_point.extend([(z, p[1]) for n, p, z in points])
        # B_point.extend([(z, p[2]) for n, p, z in points])
    return np.array(R_point), np.array(G_point), np.array(B_point)


def calculate_backscatter_values(BS_points, depth_image, iterations=10, max_mean_loss_fraction=0.1):
    # print('depth_1:',depth_image.shape)
    BS_depth_val, BS_image_val = BS_points[:, 0], BS_points[:, 1]
    z_max, z_min = np.max(depth_image), np.min(depth_image)
    max_mean_loss = max_mean_loss_fraction * (z_max - z_min)
    coeffs = None
    min_loss = np.inf
    lower_limit = [0,0,0,0]
    upper_limit = [1,5,1,5]

    def estimate_coeffs(depth_image, B_inf, beta_B, J_c, beta_D):
        img_val = (B_inf * (1 - np.exp(-1 * beta_B * depth_image))) + (J_c * np.exp(-1 * beta_D * depth_image))
        # print('image val: ',img_val.shape)
        return img_val
    def estimate_loss(B_inf, beta_B, J_c, beta_D):
        loss_val = np.mean(np.abs(BS_image_val - estimate_coeffs(BS_depth_val, B_inf, beta_B, J_c, beta_D)))
        return loss_val
    
    for i in range(iterations):
        try:
            est_coeffs, _ = sp.optimize.curve_fit(
                f=estimate_coeffs,
                xdata=BS_depth_val,
                ydata=BS_image_val,
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
        print('Applying linear model.', flush=True)
        m, b, _, _, _ = sp.stats.linregress(BS_depth_val, BS_image_val)
        y = (m * depth_image) + b
        return y, np.array([m, b])
    # print("depth image 2 shape:", depth_image.shape)
    # print("estimate: ", estimate_coeffs(depth_image, *coeffs).shape)
    return estimate_coeffs(depth_image, *coeffs), coeffs

def find_closest_label(new_map, begin_x, begin_y):
    t = collections.deque()
    t.append((begin_x, begin_y))
    n_map_mask = np.zeros_like(new_map).astype(np.bool_)
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

def calculate_wideband_attentuation(depth_image, lumen, r=6, max_value=10.0):
    epsilon = 1E-8
    Beta_D = np.minimum(max_value, -np.log(lumen + epsilon) / (np.maximum(0, depth_image) + epsilon))
    masking = np.where(np.logical_and(depth_image > epsilon, lumen > epsilon), 1, 0)
    refined_atts = denoise_bilateral(closing(np.maximum(0, Beta_D * masking), disk(r)))
    return refined_atts, []

def find_beta_D(depth_image, e, f, g, h):
    return (e * np.exp(f * depth_image)) + (g * np.exp(h * depth_image))

def refining_wideband_attentuation(depth_img, illumination, estimation, iterations=10, min_depth_fraction = 0.1, max_mean_loss_fraction=np.inf, l=1.0, radius_fraction=0.01):
    epsilon = 1E-8
    z_max, z_min = np.max(depth_img), np.min(depth_img)
    min_depth = z_min + (min_depth_fraction * (z_max - z_min))
    max_mean_loss = max_mean_loss_fraction * (z_max - z_min)
    coeffs = None
    min_loss = np.inf
    lower_limit = [0, -100, 0, -100]
    upper_limit = [100, 0, 100, 0]
    indices = np.where(np.logical_and(illumination > 0, np.logical_and(depth_img > min_depth, estimation > epsilon)))
    # indices = np.asarray(np.logical_and(illumination > 0, np.logical_and(depth_img > min_depth, estimation > epsilon )))
    
    def calculate_new_rc_depths(depth_img, illum, a, b, c, d):
        E = 1E-5
        result = -np.log(illum + E) / (find_beta_D(depth_img, a, b, c, d) + E)
        return result
    def calculate_loss(a, b, c, d):
        return np.mean(np.abs(depth_img[indices] - calculate_new_rc_depths(depth_img[indices], illumination[indices], a, b, c, d)))
    
    dX, dY = data_filter(depth_img[indices], estimation[indices], radius_fraction)
    for _ in range(iterations):
        try:
            est_depth_val, _ = sp.optimize.curve_fit(
                f=find_beta_D,
                xdata=dX,
                ydata=dY,
                p0=np.abs(np.random.random(4)) * np.array([1., -1., 1., -1.]),
                bounds=(lower_limit, upper_limit))
            loss = calculate_loss(*est_depth_val)
            if loss < min_loss:
                min_loss = loss
                coeffs = est_depth_val
        except RuntimeError as re:
            print(re, file=sys.stderr)

    if min_loss > max_mean_loss:
        print('Warning: could not find accurate reconstruction. Switching to linear model.', flush=True)
        m, b, _, _, _ = sp.stats.linregress(depth_img[indices], estimation[indices])
        beta_d = (m * depth_img + b)
        return l * beta_d, np.array([m, b])
    print(f'Found best loss {min_loss}', flush=True)
    beta_d = l * find_beta_D(depth_img, *coeffs)
    return beta_d, coeffs

def image_scaling(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

# White balance based on top 10% average values of blue and green channel
def image_balancing_10(image):
    green_d = 1.0 / np.mean(np.sort(image[:, :, 1], axis=None)[int(round(-1 * np.size(image[:, :, 0]) * 0.1)):])
    blue_d = 1.0 / np.mean(np.sort(image[:, :, 2], axis=None)[int(round(-1 * np.size(image[:, :, 0]) * 0.1)):])
    sum = green_d + blue_d
    green_d = (green_d/sum)*2.0
    blue_d = (blue_d/sum)*2.0
    image[:, :, 0] *= (blue_d + green_d)/2
    image[:, :, 1] *= green_d
    image[:, :, 2] *= blue_d
    return image

# White balance with 'grey world' hypothesis - NOT USED HERE
def image_balancing_gray(image):
    green_d = 1.0 / np.mean(image[:, :, 1])
    blue_d = 1.0 / np.mean(image[:, :, 2])
    sum = green_d + blue_d
    green_d = (green_d/sum)*2.0
    blue_d = (blue_d/sum)*2.0
    image[:, :, 0] *= (blue_d + green_d)/2
    image[:, :, 1] *= green_d
    image[:, :, 2] *= blue_d
    return image

# Reconstruct the scene and globally white balance based the Gray World Hypothesis
def image_restoration(image,depth_image,updated_B,updated_beta_d,new_map):
    restoration = (image - updated_B) * np.exp(updated_beta_d * np.expand_dims(depth_image, axis=2))
    restoration = np.maximum(0.0, np.minimum(1.0, restoration))
    restoration[new_map == 0] = 0
    restoration = image_scaling(image_balancing_10(restoration))
    restoration[new_map == 0] = image[new_map == 0]
    return restoration

# Reconstruct the scene and globally white balance without depth map - NOT USED HERE
def image_restoration_S4(image, updated_B,total_illumination, new_map):
    epsl = 1E-8
    restoration = (image - updated_B) / (total_illumination + epsl)
    restoration = np.maximum(0.0, np.minimum(1.0, restoration))
    restoration[new_map == 0] = image[new_map == 0]
    scaled_image = image_scaling(image_balancing_gray(restoration))
    return scaled_image

def visualize_depth_map(rs_image, test=False, model=None):
    cmap = plt.cm.jet
    cmap.set_bad(color="black")
    # rs_image.astype(np.float32)
    
    if test:
        pred = model.predict(rs_image)
        fig, ax = plt.subplots(0, 1, figsize=(50, 50))
        for i in range(1):
            ax[i, 0].imshow((input[i].squeeze()))
            ax[i, 1].imshow((pred[i].squeeze()), cmap=cmap)
    return pred


# Global Parameters
min_depth = 0.1
max_depth = 1
equalization = True

if __name__ == "__main__":
    image = "/Users/shreejay/Desktop/UMD/ENPM673/Projects/Project5/code/dataset/raw-890/4_img_.png"
    # image = "/Users/shreejay/Desktop/UMD/ENPM673/Projects/Project5/code/dataset/reference-890/108_img_.png"
    resized_image, depth_image = readImage(image)    
    depth_image = depth_preprocess(depth_image,min_depth,max_depth)

    R_back,G_back,B_back = backscatter_estimation(resized_image,depth_image,fraction=0.01,min_depth_perc=min_depth)
    # print('Red backscatter: ',R_back[20:35])
    # print('Red backscatter shape: ',R_back.shape)
    # print('Blue backscatter: ',B_back[20:35])
    # print('Blue backscatter shape: ',B_back.shape)
    # print('Green backscatter: ',G_back[20:35])
    # print('Green backscatter shape: ',G_back.shape)
    BS_red, red_coeffs = calculate_backscatter_values(R_back, depth_image, iterations=25)
    BS_green, green_coeffs = calculate_backscatter_values(G_back, depth_image, iterations=25)
    BS_blue, blue_coeffs = calculate_backscatter_values(B_back, depth_image, iterations=25)
    # print('BS_red: ',BS_red.shape)
    # print('BS_red coeff: ',red_coeffs)
    # print('BS_green: ',BS_red.shape)
    # print('BS_red coeff: ',red_coeffs)

    new_map, _ = construct_neighbor_map(depth_image, 0.08)

    new_map, n = refining_neighbor_map(new_map, 50)

    Red_illumination = estimate_illumination_map(resized_image[:, :, 0], BS_red, new_map, n, p=0.5, f=2.0, max_iters=100, tol=1E-5)
    Green_illumination = estimate_illumination_map(resized_image[:, :, 1], BS_green, new_map, n, p=0.5, f=2.0, max_iters=100, tol=1E-5)
    Blue_illumination = estimate_illumination_map(resized_image[:, :, 2], BS_blue, new_map, n, p=0.5, f=2.0, max_iters=100, tol=1E-5)

    Total_illumination = np.stack([Red_illumination, Green_illumination, Blue_illumination], axis=2)
    
    Red_Beta_D, _ = calculate_wideband_attentuation(depth_image, Red_illumination)
    updated_red_beta_d, Red_coefs = refining_wideband_attentuation(depth_image, Red_illumination, Red_Beta_D, radius_fraction=0.01, l=1.0)
    
    Green_Beta_D, _ = calculate_wideband_attentuation(depth_image, Green_illumination)
    updated_green_beta_d, Green_coefs = refining_wideband_attentuation(depth_image, Green_illumination, Green_Beta_D, radius_fraction=0.01, l=1.0)
    
    Blue_Beta_D, _ = calculate_wideband_attentuation(depth_image, Blue_illumination)
    updated_blue_beta_d, Blue_coefs = refining_wideband_attentuation(depth_image, Blue_illumination, Blue_Beta_D, radius_fraction=0.01, l=1.0)

    updated_B = np.stack([BS_red,BS_green,BS_blue],axis=2)
    updated_beta_d = np.stack([updated_red_beta_d,updated_blue_beta_d,updated_green_beta_d],axis=2)
    
    output_image = image_restoration(resized_image,depth_image,updated_B,updated_beta_d,new_map)

    if equalization:
        output_image = exposure.equalize_adapthist(np.array(output_image),clip_limit=0.03)
        estimated_sigma = estimate_sigma(output_image, average_sigmas=True, channel_axis=-1)
        output_image = denoise_tv_chambolle(output_image,estimated_sigma)

    cv2.imwrite('/Users/shreejay/Desktop/UMD/ENPM673/Projects/Project5/code/results/Recovered_image.jpg',output_image)
    print('Sea-thru complete')