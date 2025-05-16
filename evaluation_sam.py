import os
import json
import numpy as np
import torch
import cv2
from scipy.optimize import linear_sum_assignment
from PIL import Image
from tqdm import tqdm
import cupy as cp


def load_binary_mask(image_path):
    image = Image.open(image_path).convert('L')
    mask = np.array(image)
    return (mask > 128).astype(np.uint8)


def calculate_iou(mask1, mask2):
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    return intersection / union if union != 0 else 0


def resize_mask(mask, target_height, target_width):
    return cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)


def evaluate_segmentation_results(base_dir, scenes, methods, output_dir):
    """
    Evaluates SAM segmentation masks against ground truth binary masks and saves IoU-matched segments.

    Args:
        base_dir (str): Base directory containing scenes.
        scenes (list[str]): List of scene folder names.
        methods (list[str]): List of SAM method subfolders to evaluate.
        output_dir (str): Output directory to store JSON evaluation results.
    """
    os.makedirs(output_dir, exist_ok=True)

    for scene in scenes:
        print(f'\nProcessing Scene: {scene}')
        scene_dir = os.path.join(base_dir, scene)
        gt_json_path = os.path.join(scene_dir, 'gsam2_gt', 'merged.json')
        gt_binary_dir = os.path.join(scene_dir, 'gt_binary_masks')

        if not os.path.exists(gt_json_path):
            print(f'GT JSON file for {scene} does not exist: {gt_json_path}, skipping.')
            continue

        with open(gt_json_path, 'r') as f:
            gt_data = json.load(f)

        for method in methods:
            print(f'\nEvaluating method: {method}')
            sam_json_path = os.path.join(scene_dir, 'sam_renders', method, 'merged.json')
            if not os.path.exists(sam_json_path):
                print(f'SAM JSON file for {scene} with method {method} does not exist: {sam_json_path}, skipping.')
                continue

            with open(sam_json_path, 'r') as f:
                sam_data = json.load(f)

            matched_segments = {}

            for image_name in tqdm(gt_data, desc=f"{scene} - {method}"):
                if image_name not in sam_data:
                    continue

                gt_binary_path = os.path.join(gt_binary_dir, image_name)
                if not os.path.exists(gt_binary_path):
                    continue

                gt_mask_binary = load_binary_mask(gt_binary_path)
                h, w = gt_mask_binary.shape
                gt_mask_binary_cp = cp.array(gt_mask_binary)

                gt_segments = gt_data[image_name]
                sam_segments = sam_data[image_name]

                gt_masks_cp = []
                for seg in gt_segments:
                    if 'mask' not in seg:
                        continue
                    mask = cp.array(seg['mask'])
                    resized = resize_mask(cp.asnumpy(mask), h, w)
                    overlap = (resized & gt_mask_binary).sum()
                    if cp.asnumpy(mask).sum() > 0:
                        overlap_pct = 100 * overlap / cp.asnumpy(mask).sum()
                        if overlap_pct >= 10:
                            gt_masks_cp.append(cp.array(resized))

                sam_masks_cp = []
                for seg in sam_segments:
                    if 'mask' not in seg:
                        continue
                    mask = cp.array(seg['mask'])
                    resized = resize_mask(cp.asnumpy(mask), h, w)
                    overlap = (resized & gt_mask_binary).sum()
                    if cp.asnumpy(mask).sum() > 0:
                        overlap_pct = 100 * overlap / cp.asnumpy(mask).sum()
                        if overlap_pct >= 10:
                            sam_masks_cp.append(cp.array(resized))

                if not gt_masks_cp or not sam_masks_cp:
                    continue

                gt_np = cp.asnumpy(cp.stack(gt_masks_cp))
                sam_np = cp.asnumpy(cp.stack(sam_masks_cp))

                gt_torch = torch.from_numpy(gt_np).float()
                sam_torch = torch.from_numpy(sam_np).float()

                iou_matrix = torch.zeros((gt_torch.shape[0], sam_torch.shape[0]))
                for i in range(gt_torch.shape[0]):
                    for j in range(sam_torch.shape[0]):
                        iou_matrix[i, j] = calculate_iou(
                            gt_torch[i].int().numpy(), sam_torch[j].int().numpy()
                        )

                row_ind, col_ind = linear_sum_assignment(iou_matrix.numpy(), maximize=True)

                matches = []
                for gt_idx, sam_idx in zip(row_ind, col_ind):
                    iou_value = iou_matrix[gt_idx, sam_idx].item()
                    if iou_value > 0:
                        matches.append({
                            'gt_idx': gt_idx,
                            'sam_idx': sam_idx,
                            'iou': iou_value
                        })

                matched_segments[image_name] = matches

            # Save output for the current scene and method
            output_path = os.path.join(output_dir, f"{scene}_{method}_results.json")
            with open(output_path, 'w') as f:
                json.dump(matched_segments, f, indent=2)
            print(f'Saved results to {output_path}')

if __name__ == "__main__":
    base_dir = '/data'
    scenes = ['backyard_deckchair', 'backyard_stroller']
    methods = ['gc', 'aura', ]
    output_dir = 'evaluation_sam_results'
    evaluate_segmentation_results(base_dir, scenes, methods, output_dir)
