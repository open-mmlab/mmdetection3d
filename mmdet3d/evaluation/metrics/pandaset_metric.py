# Copyright (c) OpenMMLab. All rights reserved.
"""PandaSet evaluation metric using nuScenes-style center-distance protocol.

Implements the official nuScenes detection evaluation:
  - Matching by 2D BEV center distance (not IoU)
  - AP averaged over distance thresholds {0.5, 1, 2, 4} meters
  - TP error metrics: ATE, ASE, AOE
  - NDS (nuScenes Detection Score)

Reference: Caesar et al., "nuScenes: A multimodal dataset for autonomous
driving", CVPR 2020.
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import print_log

from mmdet3d.registry import METRICS

DIST_THRESHOLDS = [0.5, 1.0, 2.0, 4.0]
TP_DIST_THRESHOLD = 2.0


def center_distance_2d(box_a, box_b):
    """2D Euclidean distance between box centers on the BEV plane (x, y)."""
    return np.sqrt((box_a[0] - box_b[0])**2 + (box_a[1] - box_b[1])**2)


def accumulate_ap(all_scores,
                  all_matches,
                  total_gt,
                  min_recall=0.1,
                  min_precision=0.1):
    """Compute AP using nuScenes protocol.

    Integrates the P-R curve only where both recall > min_recall and
    precision > min_precision. Returns 0 if 10% recall is never reached.

    Args:
        all_scores (np.ndarray): Confidence scores for all detections.
        all_matches (np.ndarray): Boolean, True if detection is a TP.
        total_gt (int): Total number of ground truth objects.
        min_recall (float): Minimum recall threshold.
        min_precision (float): Minimum precision threshold.

    Returns:
        float: Average Precision (0 to 1 scale).
    """
    if total_gt == 0 or len(all_scores) == 0:
        return 0.0

    sort_idx = np.argsort(-all_scores)
    all_matches = all_matches[sort_idx]

    tp_cumsum = np.cumsum(all_matches)
    fp_cumsum = np.cumsum(~all_matches)
    recall = tp_cumsum / total_gt
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)

    valid = (recall >= min_recall) & (precision >= min_precision)
    if not np.any(valid):
        return 0.0

    recall_valid = recall[valid]
    precision_valid = precision[valid]

    n_sample = 101
    recall_interp = np.linspace(min_recall, 1.0, n_sample)
    precision_interp = np.zeros(n_sample)
    for i, r in enumerate(recall_interp):
        precs = precision_valid[recall_valid >= r]
        if len(precs) > 0:
            precision_interp[i] = np.max(precs)

    ap = np.mean(precision_interp)
    return ap


def compute_tp_errors(tp_list):
    """Compute TP error metrics from matched TP pairs.

    Uses the nuScenes protocol: cumulative mean at each recall level,
    then average over recall levels above 10%.

    Args:
        tp_list: List of dicts with keys 'score', 'ate', 'ase', 'aoe'.

    Returns:
        dict: {'ate': float, 'ase': float, 'aoe': float} or None if < 10%
              recall is achieved.
    """
    if len(tp_list) == 0:
        return None

    tp_arr = sorted(tp_list, key=lambda x: -x['score'])
    ates = np.array([t['ate'] for t in tp_arr])
    ases = np.array([t['ase'] for t in tp_arr])
    aoes = np.array([t['aoe'] for t in tp_arr])

    cum_ate = np.cumsum(ates) / np.arange(1, len(ates) + 1)
    cum_ase = np.cumsum(ases) / np.arange(1, len(ases) + 1)
    cum_aoe = np.cumsum(aoes) / np.arange(1, len(aoes) + 1)

    return {
        'ate': float(np.mean(cum_ate)),
        'ase': float(np.mean(cum_ase)),
        'aoe': float(np.mean(cum_aoe)),
    }


def yaw_diff(yaw_a, yaw_b):
    """Smallest absolute angle difference between two yaw angles."""
    diff = abs(yaw_a - yaw_b)
    diff = diff % (2 * np.pi)
    if diff > np.pi:
        diff = 2 * np.pi - diff
    return diff


def compute_scale_error(dt_box, gt_box):
    """Compute ASE = 1 - IoU3D after center and orientation alignment.

    Aligns the boxes (same center, same yaw) and computes volume overlap.
    This reduces to 1 - (min(dx)/max(dx) * min(dy)/max(dy) * min(dz)/max(dz))
    for axis-aligned boxes of same center.
    """
    dt_dims = dt_box[3:6]  # dx, dy, dz
    gt_dims = gt_box[3:6]

    vol_dt = dt_dims[0] * dt_dims[1] * dt_dims[2]
    vol_gt = gt_dims[0] * gt_dims[1] * gt_dims[2]

    if vol_dt <= 0 or vol_gt <= 0:
        return 1.0

    inter = (
        min(dt_dims[0], gt_dims[0]) * min(dt_dims[1], gt_dims[1]) *
        min(dt_dims[2], gt_dims[2]))
    iou = inter / (vol_dt + vol_gt - inter)
    return 1.0 - iou


@METRICS.register_module()
class PandaSetMetric(BaseMetric):
    """PandaSet evaluation using nuScenes-style center-distance protocol.

    Computes:
      - mAP: mean AP over center-distance thresholds {0.5, 1, 2, 4}m
      - mATE: mean Average Translation Error (2D Euclidean, meters)
      - mASE: mean Average Scale Error (1 - IoU after alignment)
      - mAOE: mean Average Orientation Error (yaw angle diff, radians)
      - NDS: nuScenes Detection Score

    Args:
        ann_file (str): Path to the annotation info pkl file.
        metric (str): Unused, kept for interface compatibility.
        pcd_limit_range (list, optional): Point cloud range for GT filtering.
        collect_device (str): Device for collecting results.
        prefix (str, optional): Prefix for metric keys.
    """

    def __init__(self,
                 ann_file: str,
                 metric: str = 'bbox',
                 pcd_limit_range: Optional[list] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ann_file = ann_file
        if pcd_limit_range is not None:
            self.pcd_limit_range = pcd_limit_range
        else:
            self.pcd_limit_range = [-50, -50, -5, 50, 50, 3]

        import pickle
        with open(ann_file, 'rb') as f:
            self.data_infos = pickle.load(f)
        self.classes = list(self.data_infos['metainfo']['CLASSES'])

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process predictions and GT for one batch."""
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances_3d']
            result['bboxes_3d'] = pred['bboxes_3d'].tensor.cpu().numpy()
            result['scores_3d'] = pred['scores_3d'].cpu().numpy()
            result['labels_3d'] = pred['labels_3d'].cpu().numpy()

            eval_ann = data_sample['eval_ann_info']
            gt_bboxes = eval_ann['gt_bboxes_3d']
            if hasattr(gt_bboxes, 'tensor'):
                gt_bboxes = gt_bboxes.tensor.cpu().numpy()
            elif not isinstance(gt_bboxes, np.ndarray):
                gt_bboxes = np.array(gt_bboxes)

            gt_labels = eval_ann['gt_labels_3d']
            if hasattr(gt_labels, 'numpy'):
                gt_labels = gt_labels.numpy()

            # Filter GT to detection range
            if len(gt_bboxes) > 0 and self.pcd_limit_range is not None:
                pcr = self.pcd_limit_range
                centers = gt_bboxes[:, :3]
                mask = ((centers[:, 0] >= pcr[0]) & (centers[:, 0] <= pcr[3]) &
                        (centers[:, 1] >= pcr[1]) & (centers[:, 1] <= pcr[4]) &
                        (centers[:, 2] >= pcr[2]) & (centers[:, 2] <= pcr[5]))
                gt_bboxes = gt_bboxes[mask]
                gt_labels = gt_labels[mask]

            result['gt_bboxes_3d'] = gt_bboxes
            result['gt_labels_3d'] = gt_labels

            self.results.append(result)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute nuScenes-style metrics: mAP, mATE, mASE, mAOE, NDS."""
        print_log('\n' + '=' * 70, logger='current')
        print_log(
            'PandaSet Evaluation (nuScenes-style '
            'center-distance protocol)',
            logger='current')
        print_log('=' * 70, logger='current')

        all_dt_boxes = [r['bboxes_3d'] for r in results]
        all_dt_scores = [r['scores_3d'] for r in results]
        all_dt_labels = [r['labels_3d'] for r in results]
        all_gt_boxes = [r['gt_bboxes_3d'] for r in results]
        all_gt_labels = [r['gt_labels_3d'] for r in results]

        ap_dict = OrderedDict()
        class_aps = []
        class_tp_errors = []

        for cls_idx, cls_name in enumerate(self.classes):
            # --- AP at each distance threshold ---
            aps_for_class = []
            for dist_th in DIST_THRESHOLDS:
                ap = self._eval_class_ap(all_dt_boxes, all_dt_scores,
                                         all_dt_labels, all_gt_boxes,
                                         all_gt_labels, cls_idx, dist_th)
                aps_for_class.append(ap)
                ap_dict[f'{cls_name}/AP_dist_{dist_th}'] = ap

            mean_ap_cls = float(np.mean(aps_for_class))
            ap_dict[f'{cls_name}/AP'] = mean_ap_cls
            class_aps.append(mean_ap_cls)

            # --- TP errors at 2m threshold ---
            tp_errors = self._eval_class_tp_errors(all_dt_boxes, all_dt_scores,
                                                   all_dt_labels, all_gt_boxes,
                                                   all_gt_labels, cls_idx,
                                                   TP_DIST_THRESHOLD)
            class_tp_errors.append(tp_errors)

            if tp_errors is not None:
                ap_dict[f'{cls_name}/ATE'] = tp_errors['ate']
                ap_dict[f'{cls_name}/ASE'] = tp_errors['ase']
                ap_dict[f'{cls_name}/AOE'] = tp_errors['aoe']
            else:
                ap_dict[f'{cls_name}/ATE'] = 1.0
                ap_dict[f'{cls_name}/ASE'] = 1.0
                ap_dict[f'{cls_name}/AOE'] = 1.0

        # --- Aggregate metrics ---
        mAP = float(np.mean(class_aps))

        ate_vals = [
            e['ate'] if e is not None else 1.0 for e in class_tp_errors
        ]
        ase_vals = [
            e['ase'] if e is not None else 1.0 for e in class_tp_errors
        ]
        aoe_vals = [
            e['aoe'] if e is not None else 1.0 for e in class_tp_errors
        ]

        mATE = float(np.mean(ate_vals))
        mASE = float(np.mean(ase_vals))
        mAOE = float(np.mean(aoe_vals))

        # NDS with 5 TP metrics (AVE=1, AAE=1 since unavailable)
        tp_scores = [
            max(1.0 - mATE, 0.0),
            max(1.0 - mASE, 0.0),
            max(1.0 - mAOE, 0.0),
            0.0,  # 1 - min(1, mAVE=1) = 0
            0.0,  # 1 - min(1, mAAE=1) = 0
        ]
        nds = (5.0 * mAP + sum(tp_scores)) / 10.0

        # NDS partial (only 3 available TP metrics)
        tp_scores_partial = tp_scores[:3]
        nds_partial = (5.0 * mAP + sum(tp_scores_partial)) / 8.0

        ap_dict['mAP'] = mAP
        ap_dict['mATE'] = mATE
        ap_dict['mASE'] = mASE
        ap_dict['mAOE'] = mAOE
        ap_dict['NDS'] = nds
        ap_dict['NDS_partial'] = nds_partial

        # --- Print formatted results ---
        self._print_results(class_aps, class_tp_errors, mAP, mATE, mASE, mAOE,
                            nds, nds_partial)

        # --- Distance-bin analysis ---
        dist_bin_results = self._compute_distance_bin_analysis(
            all_dt_boxes, all_dt_scores, all_dt_labels, all_gt_boxes,
            all_gt_labels)
        ap_dict.update(dist_bin_results)

        return ap_dict

    def _eval_class_ap(self, all_dt_boxes, all_dt_scores, all_dt_labels,
                       all_gt_boxes, all_gt_labels, class_id, dist_threshold):
        """Compute AP for one class at one center-distance threshold."""
        all_scores = []
        all_matches = []
        total_gt = 0

        for dt_boxes, dt_scores, dt_labels, gt_boxes, gt_labels in zip(
                all_dt_boxes, all_dt_scores, all_dt_labels, all_gt_boxes,
                all_gt_labels):

            dt_mask = dt_labels == class_id
            gt_mask = gt_labels == class_id

            dt_cls_boxes = dt_boxes[dt_mask]
            dt_cls_scores = dt_scores[dt_mask]
            gt_cls_boxes = gt_boxes[gt_mask]

            num_gt = len(gt_cls_boxes)
            total_gt += num_gt

            if len(dt_cls_boxes) == 0:
                continue

            if num_gt == 0:
                all_scores.extend(dt_cls_scores.tolist())
                all_matches.extend([False] * len(dt_cls_scores))
                continue

            # Compute BEV center distances (N_dt x N_gt)
            dt_centers = dt_cls_boxes[:, :2]
            gt_centers = gt_cls_boxes[:, :2]
            dist_matrix = np.sqrt(
                ((dt_centers[:, None, :] - gt_centers[None, :, :])**2).sum(-1))

            # Greedy matching: sort detections by score, assign closest GT
            sort_idx = np.argsort(-dt_cls_scores)
            gt_matched = np.zeros(num_gt, dtype=bool)

            for idx in sort_idx:
                dists = dist_matrix[idx]
                # Mask already-matched GT
                dists_masked = dists.copy()
                dists_masked[gt_matched] = np.inf
                best_gt = np.argmin(dists_masked)

                if dists_masked[best_gt] <= dist_threshold:
                    gt_matched[best_gt] = True
                    all_scores.append(dt_cls_scores[idx])
                    all_matches.append(True)
                else:
                    all_scores.append(dt_cls_scores[idx])
                    all_matches.append(False)

        all_scores = np.array(all_scores)
        all_matches = np.array(all_matches, dtype=bool)

        return accumulate_ap(all_scores, all_matches, total_gt)

    def _eval_class_tp_errors(self, all_dt_boxes, all_dt_scores, all_dt_labels,
                              all_gt_boxes, all_gt_labels, class_id,
                              dist_threshold):
        """Compute TP error metrics (ATE, ASE, AOE) for one class at 2m."""
        tp_list = []
        total_gt = 0

        for dt_boxes, dt_scores, dt_labels, gt_boxes, gt_labels in zip(
                all_dt_boxes, all_dt_scores, all_dt_labels, all_gt_boxes,
                all_gt_labels):

            dt_mask = dt_labels == class_id
            gt_mask = gt_labels == class_id

            dt_cls_boxes = dt_boxes[dt_mask]
            dt_cls_scores = dt_scores[dt_mask]
            gt_cls_boxes = gt_boxes[gt_mask]

            num_gt = len(gt_cls_boxes)
            total_gt += num_gt

            if len(dt_cls_boxes) == 0 or num_gt == 0:
                continue

            dt_centers = dt_cls_boxes[:, :2]
            gt_centers = gt_cls_boxes[:, :2]
            dist_matrix = np.sqrt(
                ((dt_centers[:, None, :] - gt_centers[None, :, :])**2).sum(-1))

            sort_idx = np.argsort(-dt_cls_scores)
            gt_matched = np.zeros(num_gt, dtype=bool)

            for idx in sort_idx:
                dists = dist_matrix[idx]
                dists_masked = dists.copy()
                dists_masked[gt_matched] = np.inf
                best_gt = np.argmin(dists_masked)

                if dists_masked[best_gt] <= dist_threshold:
                    gt_matched[best_gt] = True
                    dt_box = dt_cls_boxes[idx]
                    gt_box = gt_cls_boxes[best_gt]

                    ate = float(dists[best_gt])
                    ase = compute_scale_error(dt_box[:7], gt_box[:7])

                    dt_yaw = dt_box[6] if dt_box.shape[0] > 6 else 0.0
                    gt_yaw = gt_box[6] if gt_box.shape[0] > 6 else 0.0
                    aoe = yaw_diff(dt_yaw, gt_yaw)

                    tp_list.append({
                        'score': float(dt_cls_scores[idx]),
                        'ate': ate,
                        'ase': ase,
                        'aoe': aoe,
                    })

        if total_gt == 0:
            return None

        # Check if at least 10% recall is achieved
        if len(tp_list) / max(total_gt, 1) < 0.1:
            return None

        return compute_tp_errors(tp_list)

    def _compute_distance_bin_analysis(self, all_dt_boxes, all_dt_scores,
                                       all_dt_labels, all_gt_boxes,
                                       all_gt_labels):
        """Compute per-distance-bin AP for each class.

        Bins: 0-25m (near-range), 25-50m (far-range).
        Uses center-distance matching averaged over {0.5, 1, 2, 4}m thresholds.
        """
        dist_bins = [(0, 25), (25, 50)]
        bin_labels = ['0-25m', '25-50m']
        ap_dict = OrderedDict()

        print_log('\n' + '=' * 70, logger='current')
        print_log(
            'Distance-Bin Analysis (AP by BEV range from ego)',
            logger='current')
        print_log('=' * 70, logger='current')

        header = f'\n{"Class":<25} '
        for bl in bin_labels:
            header += f'{bl + " AP":<12} {bl + " GT":<10} '
        header += f'{"Full AP":<10}'
        print_log(header, logger='current')
        print_log('-' * 80, logger='current')

        any_printed = False
        for cls_idx, cls_name in enumerate(self.classes):
            # Count total GT in range for this class
            total_gt_all = sum(
                int((gt_labels == cls_idx).sum())
                for gt_labels in all_gt_labels)
            if total_gt_all < 10:
                continue

            row = f'{cls_name:<25} '
            for bin_idx, (rmin, rmax) in enumerate(dist_bins):
                bin_aps = []
                bin_gt_count = 0
                for dist_th in DIST_THRESHOLDS:
                    ap, n_gt = self._eval_class_ap_range(
                        all_dt_boxes, all_dt_scores, all_dt_labels,
                        all_gt_boxes, all_gt_labels, cls_idx, dist_th, rmin,
                        rmax)
                    bin_aps.append(ap)
                    bin_gt_count = n_gt

                mean_bin_ap = float(np.mean(bin_aps))
                ap_dict[f'{cls_name}/AP_{bin_labels[bin_idx]}'] = mean_bin_ap
                ap_dict[f'{cls_name}/GT_{bin_labels[bin_idx]}'] = bin_gt_count

                if bin_gt_count > 0:
                    row += f'{mean_bin_ap:<12.4f} {bin_gt_count:<10} '
                else:
                    row += f'{"N/A":<12} {bin_gt_count:<10} '

            # Full range AP (already computed, recompute for consistency)
            full_aps = []
            for dist_th in DIST_THRESHOLDS:
                ap, _ = self._eval_class_ap_range(all_dt_boxes, all_dt_scores,
                                                  all_dt_labels, all_gt_boxes,
                                                  all_gt_labels, cls_idx,
                                                  dist_th, 0, 50)
                full_aps.append(ap)
            full_ap = float(np.mean(full_aps))
            row += f'{full_ap:<10.4f}'

            print_log(row, logger='current')
            any_printed = True

        if not any_printed:
            print_log(
                '  (No classes with >= 10 GT objects in range)',
                logger='current')

        print_log('-' * 80 + '\n', logger='current')
        return ap_dict

    def _eval_class_ap_range(self, all_dt_boxes, all_dt_scores, all_dt_labels,
                             all_gt_boxes, all_gt_labels, class_id,
                             dist_threshold, range_min, range_max):
        """Compute AP for one class at one distance threshold, filtered by BEV
        distance range.

        Both GT and predictions are filtered to [range_min, range_max) BEV
        distance from ego before matching.

        Returns:
            tuple: (ap, n_gt) where n_gt is GT count in this range.
        """
        all_scores = []
        all_matches = []
        total_gt = 0

        for dt_boxes, dt_scores, dt_labels, gt_boxes, gt_labels in zip(
                all_dt_boxes, all_dt_scores, all_dt_labels, all_gt_boxes,
                all_gt_labels):

            dt_mask = dt_labels == class_id
            gt_mask = gt_labels == class_id

            dt_cls_boxes = dt_boxes[dt_mask]
            dt_cls_scores = dt_scores[dt_mask]
            gt_cls_boxes = gt_boxes[gt_mask]

            # Filter GT by BEV distance
            if len(gt_cls_boxes) > 0:
                gt_dists = np.sqrt(gt_cls_boxes[:, 0]**2 +
                                   gt_cls_boxes[:, 1]**2)
                gt_range_mask = ((gt_dists >= range_min)
                                 & (gt_dists < range_max))
                gt_cls_boxes = gt_cls_boxes[gt_range_mask]

            # Filter predictions by BEV distance
            if len(dt_cls_boxes) > 0:
                dt_dists = np.sqrt(dt_cls_boxes[:, 0]**2 +
                                   dt_cls_boxes[:, 1]**2)
                dt_range_mask = ((dt_dists >= range_min)
                                 & (dt_dists < range_max))
                dt_cls_boxes = dt_cls_boxes[dt_range_mask]
                dt_cls_scores = dt_cls_scores[dt_range_mask]

            num_gt = len(gt_cls_boxes)
            total_gt += num_gt

            if len(dt_cls_boxes) == 0:
                continue
            if num_gt == 0:
                all_scores.extend(dt_cls_scores.tolist())
                all_matches.extend([False] * len(dt_cls_scores))
                continue

            dt_centers = dt_cls_boxes[:, :2]
            gt_centers = gt_cls_boxes[:, :2]
            dist_matrix = np.sqrt(
                ((dt_centers[:, None, :] - gt_centers[None, :, :])**2).sum(-1))

            sort_idx = np.argsort(-dt_cls_scores)
            gt_matched = np.zeros(num_gt, dtype=bool)

            for idx in sort_idx:
                dists_masked = dist_matrix[idx].copy()
                dists_masked[gt_matched] = np.inf
                best_gt = np.argmin(dists_masked)

                if dists_masked[best_gt] <= dist_threshold:
                    gt_matched[best_gt] = True
                    all_scores.append(dt_cls_scores[idx])
                    all_matches.append(True)
                else:
                    all_scores.append(dt_cls_scores[idx])
                    all_matches.append(False)

        all_scores = np.array(all_scores) if all_scores else np.array([])
        all_matches = np.array(all_matches, dtype=bool) if all_matches \
            else np.array([], dtype=bool)

        ap = accumulate_ap(all_scores, all_matches, total_gt)
        return ap, total_gt

    def _print_results(self, class_aps, class_tp_errors, mAP, mATE, mASE, mAOE,
                       nds, nds_partial):
        """Print results in nuScenes-style format."""
        sep = '-' * 70
        result_str = f'\n{sep}\n'
        result_str += (f'{"Object Class":<25} {"AP":<8} {"ATE":<8} '
                       f'{"ASE":<8} {"AOE":<8}\n')
        result_str += f'{sep}\n'

        for cls_idx, cls_name in enumerate(self.classes):
            ap_str = f'{class_aps[cls_idx]:.4f}'
            tp = class_tp_errors[cls_idx]
            if tp is not None:
                ate_str = f'{tp["ate"]:.4f}'
                ase_str = f'{tp["ase"]:.4f}'
                aoe_str = f'{tp["aoe"]:.4f}'
            else:
                ate_str = 'N/A'
                ase_str = 'N/A'
                aoe_str = 'N/A'
            result_str += (f'{cls_name:<25} {ap_str:<8} {ate_str:<8} '
                           f'{ase_str:<8} {aoe_str:<8}\n')

        result_str += f'{sep}\n'
        result_str += f'mAP:  {mAP:.4f}\n'
        result_str += f'mATE: {mATE:.4f}\n'
        result_str += f'mASE: {mASE:.4f}\n'
        result_str += f'mAOE: {mAOE:.4f}\n'
        result_str += 'mAVE: N/A (no velocity in PandaSet)\n'
        result_str += 'mAAE: N/A (no attributes in PandaSet)\n'
        result_str += f'NDS:  {nds:.4f} (with mAVE=1, mAAE=1)\n'
        result_str += f'NDS (partial, 3 TP metrics only): {nds_partial:.4f}\n'
        result_str += f'{sep}\n'

        print_log(result_str, logger='current')
