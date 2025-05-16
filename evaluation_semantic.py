import os
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def calculate_iou(reference_mask, pred_mask):
    intersection = torch.logical_and(reference_mask, pred_mask).sum().item()
    union = torch.logical_or(reference_mask, pred_mask).sum().item()
    return intersection / union if union > 0 else 0.0


def calculate_per_pixel_accuracy(reference_mask, real_mask, pred_mask):
    pred_mask = pred_mask * reference_mask
    correct_pixels = torch.sum((pred_mask == real_mask) & (reference_mask > 0)).item()
    total_pixels = torch.sum(reference_mask > 0).item()
    return correct_pixels / total_pixels if total_pixels > 0 else 0.0


def resize_to_largest(*masks):
    max_h = max(mask.shape[0] for mask in masks)
    max_w = max(mask.shape[1] for mask in masks)

    resized = []
    for mask in masks:
        # Use nearest interpolation for masks
        tensor_mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        resized_mask = F.interpolate(tensor_mask, size=(max_h, max_w), mode='nearest').squeeze()
        resized.append(resized_mask > 0)
    return resized


def evaluate_gsam_masks(input_path, method):
    """
    Evaluate GSAM masks for a given directory and method.
    Expects this directory structure:
    - {input_path}/gsam2_renders/{method}/mask/
    - {input_path}/gsam2_before/{method}/mask/
    - {input_path}/gsam2_gt/mask/
    
    Saves results as JSON in {input_path}/global_evaluation_g_sam/
    """
    output_path = os.path.join(input_path, 'global_evaluation_g_sam')
    os.makedirs(output_path, exist_ok=True)

    gsam_after_dir = os.path.join(input_path, f'gsam2_renders/{method}/mask')
    gsam_before_dir = os.path.join(input_path, f'gsam2_before/{method}/mask')
    gt_dir = os.path.join(input_path, 'gsam2_gt/mask')

    image_files = [f for f in os.listdir(gt_dir) if f.endswith('.png')]
    results = []

    for img_name in image_files:
        ref_path = os.path.join(gt_dir, img_name)
        before_path = os.path.join(gsam_before_dir, img_name)
        after_path = os.path.join(gsam_after_dir, img_name)

        # Load masks (fallback to zero mask if missing)
        ref_mask = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        if ref_mask is None:
            print(f"Reference mask missing: {ref_path}, skipping.")
            continue

        before_mask = cv2.imread(before_path, cv2.IMREAD_GRAYSCALE)
        after_mask = cv2.imread(after_path, cv2.IMREAD_GRAYSCALE)

        if before_mask is None:
            before_mask = np.zeros_like(ref_mask)
        if after_mask is None:
            after_mask = np.zeros_like(ref_mask)

        # Resize masks to the largest shape
        ref_mask_t, before_mask_t, after_mask_t = resize_to_largest(ref_mask, before_mask, after_mask)

        # Calculate IoUs and accuracies
        iou_before = calculate_iou(ref_mask_t, before_mask_t)
        iou_after = calculate_iou(ref_mask_t, after_mask_t)

        acc_before = calculate_per_pixel_accuracy(ref_mask_t, ref_mask_t, before_mask_t)
        acc_after = calculate_per_pixel_accuracy(ref_mask_t, torch.zeros_like(ref_mask_t), after_mask_t)

        results.append({
            'image_name': img_name,
            'iou_before': iou_before,
            'iou_after': iou_after,
            'iou_diff': iou_after - iou_before,
            'acc_before': acc_before,
            'acc_after': acc_after,
            'acc_diff': acc_after - acc_before,
        })

        print(f"{img_name}: IoU diff {iou_after - iou_before:.4f}, Acc diff {acc_after - acc_before:.4f}")

    # Save results
    with open(os.path.join(output_path, f'evaluation_{method}.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate GSAM masks for a given path and method')
    parser.add_argument('input_path', type=str, help='Input directory path')
    parser.add_argument('method', type=str, help='Method name')
    args = parser.parse_args()

    evaluate_gsam_masks(args.input_path, args.method)


import json
import os
import argparse

def main(source_dirs, methods, results_file_path):
    with open(results_file_path, 'w') as results_file:
        for method in methods:
            for source_dir in source_dirs:
                scene = os.path.basename(source_dir.rstrip('/'))
                json_path = os.path.join(source_dir, 'global_evaluation_g_sam', f'all_results_{method}.json')

                if not os.path.isfile(json_path):
                    print(f"Warning: JSON file not found: {json_path}")
                    continue

                with open(json_path, 'r') as file:
                    data = json.load(file)

                iou_before_sum = iou_after_sum = iou_diff_sum = 0.0
                acc_before_sum = acc_after_sum = acc_diff_sum = 0.0
                count_iou_after_05 = 0
                count = len(data)

                for entry in data:
                    iou_before_sum += entry.get('iou_before', 0)
                    iou_after_sum += entry.get('iou_after', 0)
                    iou_diff_sum += entry.get('iou_diff', 0)

                    acc_before_sum += entry.get('acc_before', 0)
                    acc_after_sum += entry.get('acc_after', 0)
                    acc_diff_sum += entry.get('acc_diff', 0)

                    if entry.get('iou_after', 0) > 0.5:
                        count_iou_after_05 += 1

                iou_before_mean = iou_before_sum / count if count else 0
                iou_after_mean = iou_after_sum / count if count else 0
                iou_diff_mean = iou_diff_sum / count if count else 0

                acc_before_mean = acc_before_sum / count if count else 0
                acc_after_mean = acc_after_sum / count if count else 0
                acc_diff_mean = acc_diff_sum / count if count else 0

                results_file.write(f'METHOD: {method}\n')
                results_file.write(f'SCENE: {scene}\n')
                results_file.write(f'Total number of images: {count}\n')
                results_file.write(f'Mean IOU Diff: {iou_diff_mean:.2f}\n')
                results_file.write(f'Mean IOU Before: {iou_before_mean:.2f}\n')
                results_file.write(f'Mean IOU After: {iou_after_mean:.2f}\n')
                results_file.write(f'Fraction of images with IOU after > 0.5: {count_iou_after_05 / count:.2f}\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate JSON results and write summary.")
    parser.add_argument('--source_dirs', nargs='+', required=True, help='List of source directories')
    parser.add_argument('--methods', nargs='+', required=True, help='List of methods to evaluate')
    parser.add_argument('--output', required=True, help='Output results file path')

    args = parser.parse_args()
    main(args.source_dirs, args.methods, args.output)
