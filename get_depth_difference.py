import numpy as np
import json
import matplotlib.pyplot as plt
import io
from PIL import Image
import os

# Function to calculate generalized histogram thresholding
def GHT(n, x=None, nu=0, tau=0, kappa=0, omega=0.5, prelim=None):
    assert nu >= 0 and tau >= 0 and kappa >= 0 and 0 <= omega <= 1
    x, w0, w1, p0, p1, _, _, d0, d1 = prelim or preliminaries(n, x)
    v0 = clip((p0 * nu * tau**2 + d0) / (p0 * nu + w0))
    v1 = clip((p1 * nu * tau**2 + d1) / (p1 * nu + w1))
    f0 = -d0 / v0 - w0 * np.log(v0) + 2 * (w0 + kappa * omega) * np.log(w0)
    f1 = -d1 / v1 - w1 * np.log(v1) + 2 * (w1 + kappa * (1 - omega)) * np.log(w1)
    return argmax(x, f0 + f1), f0 + f1

csum = lambda z: np.cumsum(z)[:-1]
dsum = lambda z: np.cumsum(z[::-1])[-2::-1]
argmax = lambda x, f: np.mean(x[:-1][f == np.max(f)])
clip = lambda z: np.maximum(1e-30, z)

def preliminaries(n, x):
    x = np.arange(len(n), dtype=n.dtype) if x is None else x
    w0 = clip(csum(n))
    w1 = clip(dsum(n))
    p0 = w0 / (w0 + w1)
    p1 = w1 / (w0 + w1)
    mu0 = csum(n * x) / w0
    mu1 = dsum(n * x) / w1
    d0 = csum(n * x**2) - w0 * mu0**2
    d1 = dsum(n * x**2) - w1 * mu1**2
    return x, w0, w1, p0, p1, mu0, mu1, d0, d1

def im2hist(im):
    max_val = np.iinfo(im.dtype).max
    x = np.arange(max_val + 1)
    e = np.arange(-0.5, max_val + 1.5)
    im_bw = np.amax(im[...,:3], -1) if im.ndim == 3 else im
    n = np.histogram(im_bw, e)[0]
    return n, x, im_bw

def process_depth_pair(depth_before_path, depth_after_path, save_path):
    os.makedirs(save_path, exist_ok=True)
    depth1 = np.array(Image.open(depth_before_path).convert('L'))
    depth2 = np.array(Image.open(depth_after_path).convert('L'))
    imgDimY, imgDimX = depth1.shape

    diff = depth2 - depth1
    diff[(diff == depth2) | (diff == -depth1)] = 0
    diff_abs = np.abs(diff)

    prefix = os.path.splitext(os.path.basename(depth_before_path))[0]
    interm_path = os.path.join(save_path, f"{prefix}_interm.png")
    plt.imsave(interm_path, diff_abs, cmap=plt.cm.gray_r)

    im = np.array(Image.open(interm_path))
    n, x, im_bw = im2hist(im)
    prelim = preliminaries(n, x)

    t, _ = GHT(n, x, 1000, 300, 0.00, 0.00, prelim)
    mask = im_bw < t
    depthC = np.minimum(depth1, depth2)

    result = np.zeros_like(depthC)
    result[mask] = depthC[mask]

    plt.imsave(os.path.join(save_path, f"{prefix}_final_threshold.png"), mask, cmap=plt.cm.gray_r)


def batch_process_depths(depth_before_dir, depth_after_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    before_files = sorted([f for f in os.listdir(depth_before_dir) if f.endswith('.png')])
    after_files = sorted([f for f in os.listdir(depth_after_dir) if f.endswith('.png')])
    common_files = set(before_files) & set(after_files)

    for filename in common_files:
        before_path = os.path.join(depth_before_dir, filename)
        after_path = os.path.join(depth_after_dir, filename)
        process_depth_pair(before_path, after_path, save_dir)


# Example usage
if __name__ == "__main__":
    depth_before_dir = "/path/to/before/depth_images"
    depth_after_dir = "/path/to/after/depth_images"
    output_dir = "/path/to/save/results"
    batch_process_depths(depth_before_dir, depth_after_dir, output_dir)
