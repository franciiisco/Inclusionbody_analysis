"""
Post-watershed hierarchical merge for correcting over-segmentation of bifid bacteria.

Over-segmentation occurs when watershed splits branching cells (Y/V shapes) into
multiple segments. This module analyzes adjacent segment pairs using multiple
features and merges those that likely belong to the same cell.
"""

import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from skimage import measure, segmentation
from skimage.morphology import skeletonize
from scipy import ndimage as ndi
from typing import Dict, List, Set, Tuple


def _find_adjacent_pairs(labels: np.ndarray) -> Set[Tuple[int, int]]:
    """Find all pairs of adjacent label IDs (excluding background 0). Fully vectorized."""
    h_mask = labels[:, :-1] != labels[:, 1:]
    left = labels[:, :-1][h_mask]
    right = labels[:, 1:][h_mask]

    v_mask = labels[:-1, :] != labels[1:, :]
    top = labels[:-1, :][v_mask]
    bottom = labels[1:, :][v_mask]

    a = np.concatenate([left, top])
    b = np.concatenate([right, bottom])

    valid = (a != 0) & (b != 0)
    a, b = a[valid], b[valid]

    if len(a) == 0:
        return set()

    pairs_array = np.column_stack([np.minimum(a, b), np.maximum(a, b)])
    unique_pairs = set(map(tuple, np.unique(pairs_array, axis=0)))
    return unique_pairs


def _combined_bbox(bbox_u, bbox_v, image_shape, padding=2):
    """Compute the combined bounding box of two regions with padding."""
    min_row = max(0, min(bbox_u[0], bbox_v[0]) - padding)
    min_col = max(0, min(bbox_u[1], bbox_v[1]) - padding)
    max_row = min(image_shape[0], max(bbox_u[2], bbox_v[2]) + padding)
    max_col = min(image_shape[1], max(bbox_u[3], bbox_v[3]) + padding)
    return min_row, min_col, max_row, max_col


def compute_merge_features(
    labels: np.ndarray,
    u: int,
    v: int,
    distance_map: np.ndarray,
    original_image: np.ndarray,
    skeleton: np.ndarray,
    bbox_u: tuple,
    bbox_v: tuple
) -> Dict[str, float]:
    """
    Extract features for a pair of adjacent segments to determine merge likelihood.
    Operates on a cropped region (bounding box) for performance.
    """
    r0, c0, r1, c1 = _combined_bbox(bbox_u, bbox_v, labels.shape)

    labels_crop = labels[r0:r1, c0:c1]
    dist_crop = distance_map[r0:r1, c0:c1]
    img_crop = original_image[r0:r1, c0:c1]
    skel_crop = skeleton[r0:r1, c0:c1]

    mask_u = labels_crop == u
    mask_v = labels_crop == v
    mask_combined = mask_u | mask_v

    boundary_all = segmentation.find_boundaries(labels_crop, mode='inner')
    dilated_u = ndi.binary_dilation(mask_u, iterations=1)
    dilated_v = ndi.binary_dilation(mask_v, iterations=1)
    boundary_between = boundary_all & dilated_u & dilated_v

    features = {}

    # --- Valley depth ---
    if boundary_between.any():
        d_boundary = dist_crop[boundary_between].mean()
    else:
        d_boundary = 0.0

    d_u = dist_crop[mask_u].mean() if mask_u.any() else 0.0
    d_v = dist_crop[mask_v].mean() if mask_v.any() else 0.0
    d_interior = max(d_u, d_v)

    if d_interior > 0:
        features['valley'] = d_boundary / d_interior
    else:
        features['valley'] = 0.0

    # --- Skeleton connectivity ---
    skeleton_in_combined = skel_crop & mask_combined
    if skeleton_in_combined.any():
        skeleton_labels = measure.label(skeleton_in_combined, connectivity=2)
        features['skeleton'] = 1.0 if skeleton_labels.max() == 1 else 0.2
    else:
        features['skeleton'] = 0.5

    # --- Intensity at boundary ---
    if boundary_between.any():
        intensity_boundary = img_crop[boundary_between].mean()
        intensity_interior = img_crop[mask_combined].mean()
        if intensity_interior > 0:
            features['intensity'] = min(intensity_boundary / intensity_interior, 1.0)
        else:
            features['intensity'] = 0.0
    else:
        features['intensity'] = 0.0

    # --- Boundary ratio (shared length / min perimeter) ---
    shared_length = float(boundary_between.sum())
    perimeter_u = measure.perimeter(mask_u)
    perimeter_v = measure.perimeter(mask_v)
    min_perimeter = min(perimeter_u, perimeter_v)

    if min_perimeter > 0:
        features['boundary'] = min(shared_length / min_perimeter, 1.0)
    else:
        features['boundary'] = 0.0

    # --- Solidity of combined shape ---
    combined_labeled = measure.label(mask_combined.astype(np.uint8))
    props = measure.regionprops(combined_labeled)
    features['solidity'] = props[0].solidity if props else 0.0

    return features


def compute_merge_score(features: Dict[str, float], weights: Dict[str, float]) -> float:
    """Weighted combination of features into a single merge score in [0, 1]."""
    score = 0.0
    total_weight = 0.0
    for key, weight in weights.items():
        if key in features:
            score += weight * features[key]
            total_weight += weight
    return score / total_weight if total_weight > 0 else 0.0


def _build_region_cache(labels: np.ndarray) -> Dict[int, dict]:
    """Build a cache of region properties (bbox, area) from labels."""
    cache = {}
    for region in measure.regionprops(labels):
        cache[region.label] = {
            'bbox': region.bbox,
            'area': region.area,
        }
    return cache


def _score_pair(args):
    """Score a single pair. Designed for use with ThreadPoolExecutor."""
    u, v, labels, distance_map, original_image, skeleton, bbox_u, bbox_v, weights = args
    features = compute_merge_features(
        labels, u, v, distance_map, original_image, skeleton, bbox_u, bbox_v
    )
    score = compute_merge_score(features, weights)
    return (u, v, score)


def _score_all_pairs_parallel(
    eligible_pairs: List[Tuple[int, int]],
    labels: np.ndarray,
    distance_map: np.ndarray,
    original_image: np.ndarray,
    skeleton: np.ndarray,
    region_cache: Dict[int, dict],
    weights: Dict[str, float]
) -> Dict[Tuple[int, int], float]:
    """Score all eligible pairs in parallel using threads."""
    if not eligible_pairs:
        return {}

    args_list = [
        (u, v, labels, distance_map, original_image, skeleton,
         region_cache[u]['bbox'], region_cache[v]['bbox'], weights)
        for u, v in eligible_pairs
    ]

    n_workers = min(os.cpu_count() or 4, len(args_list))

    # For very few pairs, skip threading overhead
    if n_workers <= 1 or len(args_list) <= 3:
        results = [_score_pair(a) for a in args_list]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_score_pair, args_list))

    return {(u, v): score for u, v, score in results}


def _batch_merge(
    labels: np.ndarray,
    pair_scores: Dict[Tuple[int, int], float],
    merge_threshold: float,
    region_cache: Dict[int, dict]
) -> bool:
    """
    Merge all non-conflicting pairs above threshold in one pass.
    Returns True if at least one merge was performed.
    """
    above = [(u, v, s) for (u, v), s in pair_scores.items() if s >= merge_threshold]
    if not above:
        return False

    above.sort(key=lambda x: -x[2])

    used_this_round: Set[int] = set()
    merged_any = False

    for u, v, _ in above:
        if u in used_this_round or v in used_this_round:
            continue

        # Execute merge: absorb v into u
        labels[labels == v] = u
        used_this_round.add(u)
        used_this_round.add(v)
        merged_any = True

        # Update cache incrementally
        merged_mask = labels == u
        rows, cols = np.where(merged_mask)
        region_cache[u] = {
            'bbox': (int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1),
            'area': int(merged_mask.sum()),
        }
        if v in region_cache:
            del region_cache[v]

    return merged_any


def hierarchical_merge(
    labels: np.ndarray,
    distance_map: np.ndarray,
    original_image: np.ndarray,
    binary_mask: np.ndarray,
    merge_config: Dict
) -> np.ndarray:
    """
    Iteratively merge over-segmented regions using multi-feature scoring.

    Uses batch merge (multiple non-conflicting pairs per iteration) and
    parallel scoring via threads for performance.
    Stops when no pair exceeds merge_threshold.
    """
    merge_threshold = merge_config.get('merge_threshold', 0.55)
    max_merged_area = merge_config.get('max_merged_area', 5000)
    weights = merge_config.get('weights', {
        'valley': 0.30,
        'skeleton': 0.30,
        'intensity': 0.15,
        'boundary': 0.10,
        'solidity': 0.15,
    })

    labels = labels.copy()
    skeleton = skeletonize(binary_mask > 0)
    region_cache = _build_region_cache(labels)

    # Score cache: persists across iterations, invalidated selectively
    score_cache: Dict[Tuple[int, int], float] = {}

    while True:
        adjacent_pairs = _find_adjacent_pairs(labels)
        if not adjacent_pairs:
            break

        # Filter eligible pairs (area constraint)
        eligible = []
        for u, v in adjacent_pairs:
            if u not in region_cache or v not in region_cache:
                continue
            if region_cache[u]['area'] + region_cache[v]['area'] > max_merged_area:
                continue
            eligible.append((u, v))

        if not eligible:
            break

        # Determine which pairs need fresh scoring (not in cache)
        pairs_to_score = [p for p in eligible if p not in score_cache]

        # Parallel scoring of uncached pairs
        new_scores = _score_all_pairs_parallel(
            pairs_to_score, labels, distance_map, original_image,
            skeleton, region_cache, weights
        )
        score_cache.update(new_scores)

        # Build current scores dict from cache (only eligible pairs)
        current_scores = {p: score_cache[p] for p in eligible if p in score_cache}

        # Batch merge all non-conflicting pairs above threshold
        merged = _batch_merge(labels, current_scores, merge_threshold, region_cache)

        if not merged:
            break

        # Invalidate cached scores for any pair involving a merged region
        touched = set()
        for (u, v), s in current_scores.items():
            if s >= merge_threshold:
                touched.add(u)
                touched.add(v)

        score_cache = {
            pair: s for pair, s in score_cache.items()
            if pair[0] not in touched and pair[1] not in touched
        }

    # Relabel sequentially
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]
    relabeled = np.zeros_like(labels)
    for new_id, old_id in enumerate(unique_labels, start=1):
        relabeled[labels == old_id] = new_id

    return relabeled
