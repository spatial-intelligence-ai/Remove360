import os
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F

def calculate_iou(reference_mask, pred_mask):
    pred_mask = pred_mask * reference_mask
    intersection = torch.logical_and(reference_mask, pred_mask).sum().item()
    union = torch.logical_or(reference_mask, pred_mask).sum().item()
    return intersection / union if union > 0 else 0.0

def calculate_accuracy(reference_mask, pred_mask):
    pred_mask = pred_mask * reference_mask
    correct = ((pred_mask == reference_mask) & (reference_mask > 0)).sum().item()
    total = (reference_mask > 0).sum().item()
    return correct / total if total > 0 else 0.0

def load_mask(path):
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return torch.from_numpy(img)

def match_size(ref, target):
    if ref.shape != target.shape:
        target = F.interpolate(
            target.unsqueeze(0).unsqueeze(0).float(), 
            size=ref.shape, 
            mode='nearest-exact'
        ).squeeze().to(torch.uint8)
    return target

def evaluate_scene(scene_dir):
    print(f"\nEvaluating: {scene_dir}")
    depth_path = os.path.join(scene_dir, 'depth_after_removal/aura')
    ref_path = os.path.join(scene_dir, 'gsam2_gt/mask')
    out_path = os.path.join(scene_dir, 'evaluation_depth_diff')
    os.makedirs(out_path, exist_ok=True)

    image_names = [f.replace('_final_threshold.png', '') 
                   for f in os.listdir(depth_path) 
                   if f.endswith('_final_threshold.png')]

    results = []
    for name in image_names:
        ref = load_mask(os.path.join(ref_path, f'{name}.png'))
        depth = load_mask(os.path.join(depth_path, f'{name}_final_threshold.png'))

        if depth is None:
            print(f"Missing depth: {name}")
            continue

        if ref is None:
            ref = torch.zeros_like(depth)

        depth = match_size(ref, depth)

        ref_bin = (ref > 0)
        depth_bin = (depth != 255)

        iou = calculate_iou(ref_bin, depth_bin)
        acc = calculate_accuracy(ref_bin, depth_bin)

        print(f"Image: {name} | IoU: {iou:.3f} | Acc: {acc:.3f}")
        results.append({'image': name, 'iou': iou, 'acc': acc})

    with open(os.path.join(out_path, 'depth_evaluation.json'), 'w') as f:
        json.dump(results, f)
    return results

def summarize_results(results, label="SUMMARY"):
    if not results:
        print(f"{label}: No data")
        return
    ious = [r['iou'] for r in results]
    accs = [r['acc'] for r in results]
    print(f"\n{label}:")
    print(f"Mean IoU: {np.mean(ious):.3f}")
    print(f"Mean Accuracy: {np.mean(accs):.3f}")

def evaluate_all(scenes):
    for scene in scenes:
        results = evaluate_scene(scene)
        summarize_results(results, label=os.path.basename(scene))

# Usage
scene_paths = [
    '/data/backyard_deckchair',
    '/data/backyard_stroller',
    # Add more scene paths as needed
]
evaluate_all(scene_paths)
