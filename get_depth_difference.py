import numpy as np
import json
import matplotlib.pyplot as plt
import io
from PIL import Image
import os
import pdb

# Function to calculate generalized histogram thresholding
def GHT(n, x=None, nu=0, tau=0, kappa=0, omega=0.5, prelim=None):
    assert nu >= 0
    assert tau >= 0
    assert kappa >= 0
    assert 0 <= omega <= 1
    x, w0, w1, p0, p1, _, _, d0, d1 = prelim or preliminaries(n, x)
    v0 = clip((p0 * nu * tau**2 + d0) / (p0 * nu + w0))
    v1 = clip((p1 * nu * tau**2 + d1) / (p1 * nu + w1))
    f0 = -d0 / v0 - w0 * np.log(v0) + 2 * (w0 + kappa * omega) * np.log(w0)
    f1 = -d1 / v1 - w1 * np.log(v1) + 2 * (w1 + kappa * (1 - omega)) * np.log(w1)
    return argmax(x, f0 + f1), f0 + f1

# Helper functions
csum = lambda z: np.cumsum(z)[:-1]
dsum = lambda z: np.cumsum(z[::-1])[-2::-1]
argmax = lambda x, f: np.mean(x[:-1][f == np.max(f)])  # Use the mean for ties.
clip = lambda z: np.maximum(1e-30, z)

# Function to calculate preliminary values
def preliminaries(n, x):
    assert np.all(n >= 0)
    x = np.arange(len(n), dtype=n.dtype) if x is None else x
    assert np.all(x[1:] >= x[:-1])
    w0 = clip(csum(n))
    w1 = clip(dsum(n))
    p0 = w0 / (w0 + w1)
    p1 = w1 / (w0 + w1)
    mu0 = csum(n * x) / w0
    mu1 = dsum(n * x) / w1
    d0 = csum(n * x**2) - w0 * mu0**2
    d1 = dsum(n * x**2) - w1 * mu1**2
    return x, w0, w1, p0, p1, mu0, mu1, d0, d1

# Convert image to histogram data
def im2hist(im, zero_extents=False):
    max_val = np.iinfo(im.dtype).max
    x = np.arange(max_val + 1)
    e = np.arange(-0.5, max_val + 1.5)
    im_bw = np.amax(im[...,:3], -1) if im.ndim == 3 else im
    n = np.histogram(im_bw, e)[0]
    if zero_extents:
        n[0], n[-1] = 0, 0
    return n, x, im_bw

# Load images and compute depth difference
def load_and_process_depth(depth_depth_rgb_path, depth_depth_removed_path, path_to_save, box_prior=False, box_coords=None):
    os.makedirs(path_to_save, exist_ok=True)
    depth1 = np.array(Image.open(depth_depth_rgb_path).convert('L'))  # Load PNG image and convert to grayscale
    depth = np.array(Image.open(depth_depth_removed_path).convert('L'))    # Load PNG image and convert to grayscale
    
    imgDimY, imgDimX = np.shape(depth)
    # Extract file name prefix from the path
    file_prefix = os.path.basename(depth_depth_rgb_path).split('.')[0]
    os.makedirs(os.path.join(path_to_save, 'depth_before'), exist_ok=True)
    os.makedirs(os.path.join(path_to_save, 'depth_after'), exist_ok=True)
    plt.imsave(os.path.join(path_to_save, 'depth_before', f'{file_prefix}.png'), depth1, cmap=plt.cm.gray)
    plt.imsave(os.path.join(path_to_save, 'depth_after', f'{file_prefix}.png'), depth, cmap=plt.cm.gray)
    
    
    if box_prior:
        # Create a mask for the box defined by box_coords
        x1, y1, x2, y2 = box_coords if box_coords is not None else (0, 0, imgDimX, imgDimY)  # Default to full image if None
        box_mask = np.zeros_like(depth, dtype=bool)
        box_mask[int(y1):int(y2), int(x1):int(x2)] = True  # Set the box area to True

        # Change here to set values outside the box to 0
        depth1_box = np.where(box_mask, depth1, 0)  # Keep values inside the box, set others to 0
        depth_box = np.where(box_mask, depth, 0)    # Keep values inside the box, set others to 0
        # Change here and select closer depth
        newDepth = depth_box - depth1_box
        newDepthN = newDepth
        newDepthN[newDepth == depth_box] = 0
        newDepthN[newDepth == -depth1_box] = 0
    else:
        newDepth = depth - depth1
        newDepthN = newDepth
        newDepthN[newDepth == depth] = 0
        newDepthN[newDepth == -depth1] = 0
    newDepthF = np.absolute(newDepthN)
    # Save the intermediate image with the file prefix
    plt.imsave(os.path.join(path_to_save, f'{file_prefix}_interm.png'), newDepthF, cmap=plt.cm.gray_r)

    byteImgIO = io.BytesIO()
    
    # Change here so that the function does not overwrite the file
    byteImg = Image.open(os.path.join(path_to_save, f'{file_prefix}_interm.png'))
    byteImg.save(byteImgIO, "PNG")
    byteImgIO.seek(0)
    byteImg = byteImgIO.read()
    im = np.array(Image.open(io.BytesIO(byteImg)))

    # Precompute a histogram and some integrals.
    n, x, im_bw = im2hist(im)
    prelim = preliminaries(n, x)

    default_nu = np.sum(n)
    default_tau = np.sqrt(1/12)
    default_kappa = np.sum(n)
    default_omega = 0

    _nu = default_nu
    _tau = default_tau
    _kappa = default_kappa
    _omega = default_omega

    omega = 0
    tau = 500e+2
    nu = 500e+2
    kappa = 0
    t, score = GHT(n, x, 1000, 300, 0.00, 0.00, prelim)
    F = np.where(im_bw[:, :] < t)

    mask = (im_bw < t)
    
    print(np.where(mask == 0))

    depthC = np.minimum(depth, depth1)

    new_array = np.empty((imgDimY, imgDimX))
    new_array[:] = 0
    new_array[mask] = depthC[mask]

    # Save the final threshold image with the file prefix
    plt.imsave(os.path.join(path_to_save, f'{file_prefix}_final_threshold.png'), im_bw < t, cmap=plt.cm.gray_r)
    

if __name__ == "__main__":

    scene_objects = [
        'backyard_deckchair' ,
       # 'backyard_patio_furniture' ,
        'backyard_stroller' ,
        'backyard_toy_house' ,
        'backyard_toy_truck' ,
        'backyard_white_chairs' ,
       # 'bedroom_drower' ,
        'bedroom_table' ,
       # 'kitchen_pots' ,
       # 'living_room_books' ,
       # 'living_room_chairs' ,
        'living_room_pillows' ,
        'living_room_sofa' ,
        'office_chairs' ,
       # 'office_monitors' ,
        'park_bicycle' ,
       # 'park_picnic_table/all' ,
       # 'park_picnic_table/cutlery' ,
       # 'park_picnic_table/flowers' ,
       # 'park_picnic_table/plates_cutlery' ,
       # 'playground_lego' ,
        'stairs_backpack' 
    ]

    box_prior=False
    # Assuming 'garden' is a boolean variable defined elsewhere

    # Iterate over each scene and its corresponding objects
    for scene in scene_objects:
        depth_removed_path = f'/mnt/proj3/open-31-16/simona/anything_left_dataset/data/{scene}/aura/train/ours_30000_object_removal/depth_normal'
        depth_rgb_path = f'/mnt/proj3/open-31-16/simona/anything_left_dataset/data/{scene}/aura/train/ours_30000/depth_normal'
        path_to_save = f'/mnt/proj3/open-31-16/simona/anything_left_dataset/data/{scene}/depth_after_removal/aura'

        before_files = sorted([f for f in os.listdir(depth_rgb_path) if f.startswith("before_")])      
        after_files = sorted([f for f in os.listdir(depth_removed_path) if f.startswith("before_")])

        before_keys = {f[len("before_"):] : f for f in before_files}
        after_keys = {f[len("after_"):] : f for f in after_files} 


        common_keys = sorted(set(before_files) & set(after_files))
        print('Common keys: ', common_keys)

        for key in common_keys:

            depth_before_path = os.path.join(depth_rgb_path, key)
            depth_after_path = os.path.join(depth_removed_path, key)

            box_coords = None
            x1_values = []
            y1_values = []
            x2_values = []
            y2_values = []  

            load_and_process_depth(depth_before_path, depth_after_path, path_to_save, box_prior, box_coords)
