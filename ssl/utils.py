import os
import logging
import time
from pathlib import Path
import torch
from scipy.optimize import linear_sum_assignment
import numpy as np
import math
import cv2
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.transform import Rotation as R
import random
from collections import defaultdict


def visualize(img_path, save_path, true_or_false, matches, bboxes, pts=[], scores=None):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    for k in range(len(matches)):
        idx = matches[k]
        one_bbox = bboxes[idx]
        if true_or_false[k]:
            label_color = (0, 255, 0)
        else:
            label_color = (0, 0, 255)
        color = get_color(k)
        x1 = int(one_bbox[0] * w)
        y1 = int(one_bbox[1] * h)
        x2 = int(one_bbox[2] * w)
        y2 = int(one_bbox[3] * h)
        cv2.rectangle(img, [x1, y1], [x2, y2], color=color, thickness=3)
        cv2.putText(img, str(k), [x1, y2], cv2.FONT_HERSHEY_PLAIN, 4, label_color, thickness=3)
        if scores is not None:
            cv2.putText(img, str(round(scores[k], 2)), [x1, y1 + 20], cv2.FONT_HERSHEY_PLAIN, 2, label_color, thickness=2)
        if len(pts):
            x = int(pts[k][0] * w)
            y = int(pts[k][1] * h)
            cv2.circle(img, [x, y], 4, color, -1)
            cv2.putText(img, str(k), [x, y], cv2.FONT_HERSHEY_PLAIN, 4, label_color, thickness=3)
    cv2.imwrite(save_path + '.jpg', img)


def visualize_highlights(img_path, save_path, true_or_false, matches, bboxes, pts=[], scores=None):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    black_rect = np.ones(img.shape, dtype=np.uint8) * 0
    alpha = 0.4
    for k in range(len(matches)):
        idx = matches[k]
        one_bbox = bboxes[idx]
        if true_or_false[k]:
            label_color = (0, 255, 0)
        else:
            label_color = (0, 0, 255)
        color = get_color(k)
        x1 = int(one_bbox[0] * w)
        y1 = int(one_bbox[1] * h)
        x2 = int(one_bbox[2] * w)
        y2 = int(one_bbox[3] * h)

        sub_img = img[y1:y2, x1:x2]
        res = cv2.addWeighted(img, alpha, black_rect, 1 - alpha, 1.0)
        res[y1:y2, x1:x2] = sub_img

        cv2.putText(res, str(k), [x1, y2], cv2.FONT_HERSHEY_PLAIN, 4, label_color, thickness=3)
        if scores is not None:
            cv2.putText(res, str(round(scores[k], 2)), [x1, y1 + 20], cv2.FONT_HERSHEY_PLAIN, 2, label_color, thickness=2)
        if len(pts):
            x = int(pts[k][0] * w)
            y = int(pts[k][1] * h)
            cv2.circle(img, [x, y], 4, color, -1)
            cv2.putText(img, str(k), [x, y], cv2.FONT_HERSHEY_PLAIN, 4, label_color, thickness=3)

        cv2.imwrite(save_path + '_' + str(k) + '.jpg', res)


def visualize_cluster(img_path, save_path, true_or_false, labels, bboxes, pts=[]):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    for k in range(len(labels)):
        label = labels[k]
        one_bbox = bboxes[k]
        if true_or_false[k]:
            label_color = (0, 255, 0)
        else:
            label_color = (0, 0, 255)
        color = get_color(label)
        x1 = int(one_bbox[0] * w)
        y1 = int(one_bbox[1] * h)
        x2 = int(one_bbox[2] * w)
        y2 = int(one_bbox[3] * h)
        cv2.rectangle(img, [x1, y1], [x2, y2], color=color, thickness=3)
        cv2.putText(img, str(label), [x1, y2], cv2.FONT_HERSHEY_PLAIN, 4, label_color, thickness=3)
        if len(pts):
            x = int(pts[k][0] * w)
            y = int(pts[k][1] * h)
            cv2.circle(img, [x, y], 4, color, -1)
            cv2.putText(img, str(k), [x, y], cv2.FONT_HERSHEY_PLAIN, 4, label_color, thickness=3)
    cv2.imwrite(save_path, img)


def min_max_norm(data):
    min_v, _ = data.min(dim=0, keepdim=True)
    max_v, _ = data.max(dim=0, keepdim=True)
    new_data = (data - min_v) / (max_v - min_v + 1e-6)
    return new_data


def get_dis_matrix(pred_pts, gt_pts, mode='l2'):
    pts1 = pred_pts.repeat_interleave(len(gt_pts), 0)
    pts2 = gt_pts.repeat(len(pred_pts), 1)
    if mode == 'l2':
        dis_matrix = torch.sqrt((pts1 - pts2).pow(2).sum(-1)).reshape(len(pred_pts), len(gt_pts))
    elif mode == 'l1':
        dis_matrix = (pts1 - pts2).abs().sum(-1).reshape(len(pred_pts), len(gt_pts))
    elif mode == 'iou':
        dis_matrix = 1 - bbox_iou(pts1.T, pts2.T, x1y1x2y2=True).reshape(len(pred_pts), len(gt_pts))
    elif mode == 'giou':
        dis_matrix = 1 - bbox_iou(pts1.T, pts2.T, x1y1x2y2=True, GIoU=True).reshape(len(pred_pts), len(gt_pts))
    elif mode == 'diou':
        dis_matrix = 1 - bbox_iou(pts1.T, pts2.T, x1y1x2y2=True, DIoU=False).reshape(len(pred_pts), len(gt_pts))
    return dis_matrix

def dis_matrix_normalization(dis_matrix, min_max=False):
    min_v = 0
    if min_max:
        min_v = dis_matrix.min()
    max_v = dis_matrix.max()
    norm_dis_matrix = (dis_matrix - min_v) / (max_v - min_v + 1e-6)
    return norm_dis_matrix

def cross_view_matching_evaluation(pred_pts, gt_pts, labels1, labels2, reid_fea1=None, reid_fea2=None, mode='l2', thresh=0.6, geo_dis_matrix=None, alpha=0.1):
    labels1_set = set(labels1.tolist())
    labels2_set = set(labels2.tolist())
    # id = -1 means no id label available
    if -1 in labels1_set:
        labels1_set.remove(-1)
    if -1 in labels2_set:
        labels2_set.remove(-1)
    recall_total = len(labels1_set.intersection(labels2_set))
    IPAA_total = len(labels1_set.union(labels2_set))
    correct = 0
    true_or_false = []
    IPAA_correct = 0
    result = {}
    result['recall_total'] = recall_total
    if len(pred_pts):
        dis_matrix = get_dis_matrix(pred_pts, gt_pts, mode=mode)
        norm_dis_matrix = dis_matrix_normalization(dis_matrix)
        
        if reid_fea1 is not None:
            reid_dis_matrix = get_dis_matrix(reid_fea1, reid_fea2, mode=mode)
            norm_reid_dis_matrix = dis_matrix_normalization(reid_dis_matrix)
            norm_dis_matrix = norm_dis_matrix * (1 - alpha) + norm_reid_dis_matrix * alpha
        
        norm_dis_matrix = norm_dis_matrix.cpu().detach().numpy()
        if geo_dis_matrix is not None:
            norm_geo_dis_matrix = dis_matrix_normalization(geo_dis_matrix, True)
            norm_dis_matrix = norm_dis_matrix * 0.5 + norm_geo_dis_matrix * 0.5

        matches_x, matches_y = linear_sum_assignment(norm_dis_matrix)
        matches_x = list(matches_x)
        matches_y = list(matches_y)

        score_matrix = 1 - norm_dis_matrix
    
        for k in range(len(matches_x) - 1, -1, -1):
            if len(matches_x) == 1:
                break
            if score_matrix[matches_x[k], matches_y[k]] < thresh:
                del matches_x[k]
                del matches_y[k]
        
        precision_total = len(matches_x)
        for k in range(precision_total):
            if labels1[matches_x[k]] == -1 and labels2[matches_y[k]] == -1:
                true_or_false.append(True)
                continue
            if labels1[matches_x[k]] == labels2[matches_y[k]]:
                true_or_false.append(True)
                correct += 1
            else:
                true_or_false.append(False)
        label_matrix = np.zeros((len(labels1), len(labels2)), dtype=bool)
        label_mask = np.ones((len(labels1), len(labels2)), dtype=bool)
        for i in range(len(labels1)):
            for j in range(len(labels2)):
                if labels1[i] == labels2[j]:
                    label_matrix[i, j] = True
                if labels1[i] == -1 or labels2[j] == -1:
                    label_mask[i, j] = False
        N = IPAA_total - precision_total # predicted unmatched instances
        P = precision_total
        TP = correct
        FP = precision_total - correct
        TN = 0
        for i in range(len(labels1)):
            if labels1[i] == -1:
                continue
            if i in matches_x:
                continue
            idx1 = labels1[i]
            if idx1 not in labels2:
                TN += 1
        for i in range(len(labels2)):
            if labels2[i] == -1:
                continue
            if i in matches_y:
                continue
            idx2 = labels2[i]
            if idx2 not in labels1:
                TN += 1
        IPAA_correct = (TN + correct)
        
        result['scores'] = score_matrix.flatten()[label_mask.flatten()]
        result['labels'] = label_matrix.flatten()[label_mask.flatten()]
        result['IPAA_total'] = IPAA_total
        result['IPAA_correct'] = IPAA_correct
        result['true_or_false'] = true_or_false
        result['matches_x'] = matches_x
        result['matches_y'] = matches_y
        result['matches_scores'] = score_matrix[matches_x, matches_y]
    else:
        precision_total = 0
    result['correct'] = correct
    result['precision_total'] = precision_total

    return result


def multi_view_matching_evaluation(feas, reid_feas=None, mode='l2', thresh=0.6):
    view_num = len(feas)
    cam_pairs = []
    for i in range(view_num - 1):
        for j in range(i + 1, view_num):
            cam_pairs.append((i, j))
    # Step 1: Compute pairwise similarity between views
    pairwise_similarity = {}
    for cam_pair in cam_pairs:
        view1, view2 = cam_pair
        anchor_fea = feas[view1]
        pos_fea = feas[view2]
        if len(anchor_fea) == 0 or len(pos_fea) == 0:
            continue
        dis_matrix = get_dis_matrix(anchor_fea, pos_fea, mode=mode)
        norm_dis_matrix = dis_matrix_normalization(dis_matrix)
        score_matrix = 1 - norm_dis_matrix.cpu().detach().numpy()
        pairwise_similarity[(view1, view2)] = score_matrix
        pairwise_similarity[(view2, view1)] = score_matrix.T
    
    # Step 2: Create a similarity graph where nodes are people in each view
    # Each node is connected to nodes in other views with similarity above threshold
    edges = []
    node_map = []
    total_nodes = 0
    
    for view, emb in enumerate(feas):
        node_map.extend([(view, i) for i in range(len(emb))])
        total_nodes += len(emb)
    
    # Map from nodes to graph indices
    node_idx_map = {node: idx for idx, node in enumerate(node_map)}

    for (view1, view2), sim_matrix in pairwise_similarity.items():
        for i in range(sim_matrix.shape[0]):
            for j in range(sim_matrix.shape[1]):
                if sim_matrix[i, j] > thresh:
                    node1 = node_idx_map[(view1, i)]
                    node2 = node_idx_map[(view2, j)]
                    edges.append((node1, node2, sim_matrix[i, j]))  # Add edge with similarity score
    
    # Step 3: Cluster nodes based on the similarity graph for multi-view consistency
    edge_array = np.array([[e[0], e[1]] for e in edges])
    edge_weights = np.array([e[2] for e in edges])
    
    clustering = AgglomerativeClustering(n_clusters=None, affinity="precomputed", linkage="average", distance_threshold=1 - thresh)
    similarity_matrix = np.zeros((total_nodes, total_nodes))
    
    for (node1, node2, weight) in edges:
        similarity_matrix[node1, node2] = 1 - weight  # Use 1 - similarity as distance
    
    labels = clustering.fit_predict(similarity_matrix)

    # Step 4: Extract matches from clustering labels
    final_associations = [{} for _ in range(n_views)]
    for label in np.unique(labels):
        nodes_in_cluster = np.where(labels == label)[0]
        person_group = {}
        for node in nodes_in_cluster:
            view, person_idx = node_map[node]
            person_group[view] = person_idx
        
        # Add matches for each person in this cluster
        for view, person_idx in person_group.items():
            final_associations[view][person_idx] = label
    
    return final_associations


def matching_inference(pred_pts, gt_pts, mode='l1', thresh=0.6):
    result = {}
    if len(pred_pts):
        pts1 = pred_pts.repeat_interleave(len(gt_pts), 0)
        pts2 = gt_pts.repeat(len(pred_pts), 1)
        if mode == 'l1':
            dis_matrix = (pts1 - pts2).abs().sum(-1).reshape(len(pred_pts), len(gt_pts))
        else:
            dis_matrix = torch.sqrt((pts1 - pts2).pow(2).sum(-1)).reshape(len(pred_pts), len(gt_pts))
        norm_dis_matrix = dis_matrix.cpu().detach().numpy()
        matches_x, matches_y = linear_sum_assignment(norm_dis_matrix)
        # min_v = norm_dis_matrix.min(axis=1, keepdims=True)
        # max_v = norm_dis_matrix.max(axis=1, keepdims=True)
        # min_v = norm_dis_matrix.min()
        min_v = 0
        max_v = norm_dis_matrix.max()
        norm_dis_matrix = 1 - (norm_dis_matrix - min_v) / (max_v - min_v + 1e-6)
        matches_x = list(matches_x)
        matches_y = list(matches_y)
        for k in range(len(matches_x) - 1, -1, -1):
            if len(matches_x) == 1:
                break
            if norm_dis_matrix[matches_x[k], matches_y[k]] < thresh:
                del matches_x[k]
                del matches_y[k]
        result['matches_x'] = matches_x
        result['matches_y'] = matches_y
        result['true_or_false'] = [True for _ in range(len(matches_x))]
    else:
        result['matches_x'] = []
        result['matches_y'] = []
        result['true_or_false'] = []

    return result


def scalar_clip(x, min, max):
    """
    input: scalar
    """
    if x < min:
        return min
    if x > max:
        return max
    return x


def crop_feat(img_copy, bbox, zoomout_ratio=1.0):
    """
    input: img and reuqirement on zoomout ratio
    where img_size = (max_x, max_y)
    return: a single img crop
    """
    x1, y1, x2, y2 = bbox

    img_feat = None

    if zoomout_ratio == 1.0:
        img_feat = img_copy[int(y1):int(y2+1), int(x1):int(x2+1), :]
    elif zoomout_ratio > 1:
        h = y2 - y1
        w = x2 - x1
        img_feat = img_copy[int(max(0,y1-h*(zoomout_ratio-1)/2)):int(min(max_y,y2+1+h*(zoomout_ratio-1)/2)),
            int(max(0,x1-w*(zoomout_ratio-1)/2)):int(min(max_x,x2+1+w*(zoomout_ratio-1)/2)), :]
    return img_feat


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    if isinstance(box1, np.ndarray):
        box1 = np.asarray(box1, dtype=np.float32)
        box2 = np.asarray(box2, dtype=np.float32)
    
    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    if isinstance(box1, torch.Tensor):
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    elif isinstance(box1, np.ndarray):
        inter = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)
    else:
        raise ValueError('bbox must be array or tensor')

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        if isinstance(box1, torch.Tensor):
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        else:
            cw = np.maximum(b1_x2, b2_x2) - np.minimum(b1_x1, b2_x1)  # convex (smallest enclosing box) width
            ch = np.maximum(b1_y2, b2_y2) - np.minimum(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                if isinstance(box1, torch.Tensor):
                    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                else:
                    v = (4 / math.pi ** 2) * np.power(np.arctan(w2 / h2) - np.arctan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                alpha[np.isnan(alpha)] = 0.
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


def xyxy2xywh(bboxes):
    x = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
    y = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]
    return torch.stack([x, y, w, h], dim=1)


def get_dis(fea1, fea2, reid_fea1=None, reid_fea2=None, mode='l2', percentage=1):
    if len(fea1) and len(fea2):
        if mode == 'l2':
            distances = torch.sqrt((fea1 - fea2).pow(2).sum(-1))
        elif mode == 'l1':
            distances = (fea1 - fea2).abs().sum(-1)
        if percentage == 1:
            return distances.mean(-1)
        selected_num = int(len(fea1) * percentage)
        if selected_num == 0:
            selected_num = 1
        values, indices = torch.topk(distances, selected_num, largest=False)
        return values.mean(-1)
    else:
        return 0


def get_matching_dis(fea1, fea2, reid_fea1=None, reid_fea2=None, mode='l2', thresh=0.0, adaptive=False, alpha=0.1):
    # mask = torch.where((fea1[:, 0] > 1.) | (fea1[:, 1] > 1.) | (fea1[:, 2] < 0) | (fea1[:, 3] < 0), False, True)
    # if mask.sum() != 0:
    #     fea1 = fea1[mask]
    #     fea1 = torch.clamp(fea1, min=0.0, max=1.0)
    
    if len(fea1) and len(fea2):
        dis_matrix = get_dis_matrix(fea1, fea2, mode=mode)
        norm_dis_matrix = dis_matrix_normalization(dis_matrix)
        
        if reid_fea1 is not None:
            reid_dis_matrix = get_dis_matrix(reid_fea1, reid_fea2, mode=mode)
            norm_reid_dis_matrix = dis_matrix_normalization(reid_dis_matrix)
            norm_dis_matrix = norm_dis_matrix * (1 - alpha) + norm_reid_dis_matrix * alpha
        
        matches_x, matches_y = linear_sum_assignment(norm_dis_matrix.cpu().detach().numpy())

        score_matrix = 1 - norm_dis_matrix.detach().clone()

        if adaptive:
            scores = score_matrix[matches_x, matches_y]
            best_dis = (dis_matrix[matches_x, matches_y] * scores).sum() / len(matches_x)
            return best_dis, (matches_x, matches_y)
        
        if mode == 'iou':
            mask = (dis_matrix[matches_x, matches_y] != 1).cpu().detach().numpy()
        else:
            mask = (score_matrix[matches_x, matches_y] >= thresh).cpu().detach().numpy()
        
        if len(matches_x[mask]) == 0:
            # best_dis = dis_matrix[matches_x, matches_y].sum() / len(matches_x)
            # return best_dis, (matches_x, matches_y)
            return 0, ([], [])
        else:
            best_dis = dis_matrix[matches_x[mask], matches_y[mask]].sum() / len(matches_x[mask])
            return best_dis, (matches_x[mask], matches_y[mask])
    else:
        return 0, ([], [])


def get_center_prompts(pts1, pts2, matches_x, matches_y, mode='l2'):
    match_num = len(matches_x)
    if match_num < 2:
        return [], []
    matches_x = matches_x[..., None]
    matches_y = matches_y[..., None]
    x_idx1 = matches_x.repeat(len(matches_x), 0)
    x_idx2 = np.tile(matches_x, (len(matches_x), 1))
    x_prompts = pts1[x_idx1.squeeze(-1), x_idx2.squeeze(-1)]

    y_idx1 = matches_y.repeat(len(matches_y), 0)
    y_idx2 = np.tile(matches_y, (len(matches_y), 1))
    y_prompts = pts2[y_idx1.squeeze(-1), y_idx2.squeeze(-1)]

    return x_prompts, y_prompts


def FPR_95(labels, scores):
    """
    compute FPR@95
    """
    recall_point = 0.95
    # Sort label-score tuples by the score in descending order.
    indices = np.argsort(scores)[::-1]
    sorted_labels = labels[indices]
    sorted_scores = scores[indices]
    n_match = sum(sorted_labels)
    n_thresh = recall_point * n_match
    thresh_index = np.argmax(np.cumsum(sorted_labels) >= n_thresh)
    FP = np.sum(sorted_labels[:thresh_index] == 0)
    TN = np.sum(sorted_labels[thresh_index:] == 0)
    return float(FP) / float(FP + TN)


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_logger(cfg, cfg_name, phase='train'):
    this_dir = Path(os.path.dirname(__file__))
    root_output_dir = (this_dir / '..' / cfg.OUTPUT_DIR).resolve()
    tensorboard_log_dir = (this_dir / '..' / cfg.LOG_DIR).resolve()
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.TRAIN_DATASET
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / Path(cfg_name)

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = final_output_dir / "tb_logs"
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'model' in states:
        torch.save(states, os.path.join(output_dir, 'model_epoch_'+ str(states['epoch']) + '.pth.tar'))
        torch.save(states,
                   os.path.join(output_dir, 'model_best.pth.tar'))


def convert_rvec_tvec_to_rt(rvec):
    """
    Convert rvec (rotation vector) and tvec (translation vector) to
    rotation matrix and translation vector.

    Parameters:
        rvec (numpy.ndarray): Rotation vector (3x1 or 1x3).

    Returns:
        R (numpy.ndarray): Rotation matrix (3x3).
    """
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    return R


def epipolar_soft_constraint(
        bbox_list1, bbox_list2, intrin1, intrin2, extrin1, extrin2, shape):
    """
    inputs:
    bbox list []
    instrin : (3,3) numpy
    extr: list of len 6
    """
    
    def ext_a2b(ext_a, ext_b):
        T_a2r = ext_a
        T_b2r = ext_b
        
        T_a2b = np.matmul(T_b2r, np.linalg.inv(T_a2r))

        return T_a2b

    def find_line(pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2
        d = (y2 - y1) / (x2 - x1)
        e = y1 - x1 * d
        return [-d, 1, -e]

    def find_foot(a, b, c, pt):
        x1, y1 = pt
        temp = (-1 * (a * x1 + b * y1 + c) / (a * a + b * b))
        x = temp * a + x1
        y = temp * b + y1
        return [x, y]

    def find_dist(pt1, pt2):
        return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5

    T_a2b = ext_a2b(extrin1, extrin2)

    dist_matrix = np.zeros((len(bbox_list1), len(bbox_list2)))

    for i in range(len(bbox_list1)):
        for j in range(len(bbox_list2)):
            b1x1, b1y1, b1x2, b1y2 = bbox_list1[i]
            b2x1, b2y1, b2x2, b2y2 = bbox_list2[j]
            bbox1_2dpt = ((b1x1 + b1x2) / 2, (b1y1 + b1y2) / 2)
            bbox2_2dpt = ((b2x1 + b2x2) / 2, (b2y1 + b2y2) / 2)

            # bbox 1 in camera 2
            bbox1_3dpt = np.matmul(np.linalg.inv(intrin1), np.array([*bbox1_2dpt, 1]))
            bbox1_3dpt = np.array([*bbox1_3dpt.tolist(), 1])

            bbox1_in2_3dpt = np.matmul(T_a2b, bbox1_3dpt)[:3]
            bbox1_in2_2dpt = np.matmul(intrin2, bbox1_in2_3dpt)
            bbox1_in2_2dpt = bbox1_in2_2dpt[:2] / bbox1_in2_2dpt[2]

            # camera 1 epipole in camera 2
            epipole1_3dpt = np.array([0, 0, 0, 1])
            epipole1_in2_3dpt = np.matmul(T_a2b, epipole1_3dpt)[:3]
            epipole1_in2_2dpt = np.matmul(intrin2, epipole1_in2_3dpt)
            epipole1_in2_2dpt = epipole1_in2_2dpt[:2] / epipole1_in2_2dpt[2]

            # find epipolar line
            a, b, c = find_line(bbox1_in2_2dpt, epipole1_in2_2dpt)

            foot = find_foot(a, b, c, bbox2_2dpt)
            dist = find_dist(bbox2_2dpt, foot)

            # measure distance
            dist_matrix[i, j] = dist

    # normalize by diagonal line
    diag =  np.sqrt(shape[0]**2 + shape[1]**2)
    dist_matrix = dist_matrix / diag
    return dist_matrix


def cross_view_matching_loss(pred_bboxes, gt_bboxes):
    box1 = pred_bboxes.repeat_interleave(len(gt_bboxes), 0)
    box2 = gt_bboxes.repeat(len(pred_bboxes), 1)
    giou_loss_matrix = 1 - bbox_iou(box1.T, box2.T, x1y1x2y2=True, DIoU=True).reshape(len(pred_bboxes), len(gt_bboxes))
    matches_x, matches_y = linear_sum_assignment(giou_loss_matrix.cpu().detach().numpy())
    giou_loss = giou_loss_matrix[matches_x, matches_y].sum() / len(pred_bboxes)
    return giou_loss


def cross_view_point_matching_loss(pred_pts, gt_bboxes):
    pts1 = pred_pts.repeat_interleave(len(gt_bboxes), 0)
    box2 = gt_bboxes.repeat(len(pred_pts), 1)
    x_dis = ((pts1[:, 0] - box2[:, 0]).abs() + (pts1[:, 0] - box2[:, 2]).abs() - (box2[:, 2] - box2[:, 0])) / (box2[:, 2] - box2[:, 0]) * 0.5
    y_dis = ((pts1[:, 1] - box2[:, 1]).abs() + (pts1[:, 1] - box2[:, 3]).abs() - (box2[:, 3] - box2[:, 1])) / (box2[:, 3] - box2[:, 1]) * 0.5
    dis_matrix = (x_dis + y_dis).reshape(len(pred_pts), len(gt_bboxes))
    matches_x, matches_y = linear_sum_assignment(dis_matrix.cpu().detach().numpy())
    dis_loss = dis_matrix[matches_x, matches_y].sum() / len(pred_pts)
    return dis_loss


def generate_neg_bboxes(bboxes, point=False, in_boundry=False):
    # wh_scale = 0.5
    wh_scale = 0.2
    # xy_scale = 1.
    xy_scale = 0.2
    while 1:
        neg_bboxes = torch.zeros_like(bboxes)
        w = bboxes[:, 2] - bboxes[:, 0]
        h = bboxes[:, 3] - bboxes[:, 1]
        center_x = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
        center_y = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
        if random.random() > 0.5:
            neg_w = w + w * torch.rand(len(bboxes), device=bboxes.device) * wh_scale
            neg_h = h + h * torch.rand(len(bboxes), device=bboxes.device) * wh_scale
        else:
            neg_w = w - w * torch.rand(len(bboxes), device=bboxes.device) * wh_scale
            neg_h = h - h * torch.rand(len(bboxes), device=bboxes.device) * wh_scale
        if random.random() > 0.5:
            neg_center_x = center_x + center_x * (torch.rand(len(bboxes), device=bboxes.device) * xy_scale)
        else:
            neg_center_x = center_x - center_x * (torch.rand(len(bboxes), device=bboxes.device) * xy_scale)
        if random.random() > 0.5:
            neg_center_y = center_y + center_y * (torch.rand(len(bboxes), device=bboxes.device) * xy_scale)
        else:
            neg_center_y = center_y - center_y * (torch.rand(len(bboxes), device=bboxes.device) * xy_scale)
        neg_bboxes[:, 0] = neg_center_x - neg_w * 0.5
        neg_bboxes[:, 2] = neg_center_x + neg_w * 0.5
        neg_bboxes[:, 1] = neg_center_y - neg_h * 0.5
        neg_bboxes[:, 3] = neg_center_y + neg_h * 0.5
        mask = torch.where((neg_bboxes[:, 0] > 1.) | (neg_bboxes[:, 1] > 1.) | (neg_bboxes[:, 2] < 0) | (neg_bboxes[:, 3] < 0), False, True)
        if in_boundry:
            if mask.sum() != len(neg_bboxes):
                continue
            else:
                break
        else:
            break
    neg_bboxes = neg_bboxes[mask]
    neg_bboxes = torch.clamp(neg_bboxes, min=0.0, max=1.0)

    if point:
        x = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
        y = bboxes[:, 3]
        pts = torch.stack([x, y], dim=1)
        pt_loss = cross_view_point_matching_loss(pts, neg_bboxes)
        return neg_bboxes, pt_loss * 0.8
    else:
        giou_loss = cross_view_matching_loss(bboxes, neg_bboxes)
        return neg_bboxes, giou_loss * 0.8


def merge_cross_view_associations_with_constraints(pairs):
    """
    Merges cross-view associations into multi-view associations, enforcing constraints
    that each instance can only be associated with one instance per other view.
    
    Args:
        pairs (list of tuples): A list of tuples where each tuple contains two lists.
                                Each list represents the indices of instances that correspond
                                in two different images/views.
    
    Returns:
        list of sets: A list where each set represents a multi-view association.
    """
    # Graph representation with view tracking
    graph = defaultdict(lambda: defaultdict(dict))  # graph[view][instance][other_view] = {instances}
    
    # Build the graph with constraints
    for x, y in pairs:
        if len(x) != len(y):
            raise ValueError("Mismatched pair lengths in associations")
        
        for a, b in zip(x, y):
            view_a, instance_a = a
            view_b, instance_b = b
            
            # Ensure each instance is only connected to one per view
            if view_b in graph[view_a][instance_a]:
                if graph[view_a][instance_a][view_b] != {instance_b}:
                    continue  # Ignore noisy conflicts
            
            if view_a in graph[view_b][instance_b]:
                if graph[view_b][instance_b][view_a] != {instance_a}:
                    continue  # Ignore noisy conflicts
            
            # Add the association
            graph[view_a][instance_a].setdefault(view_b, set()).add(instance_b)
            graph[view_b][instance_b].setdefault(view_a, set()).add(instance_a)
    
    # Convert graph to a multi-view association
    def dfs(node, visited, component):
        visited.add(node)
        component.add(node)
        view, instance = node
        
        for neighbor_view, neighbors in graph[view][instance].items():
            for neighbor in neighbors:
                neighbor_node = (neighbor_view, neighbor)
                if neighbor_node not in visited:
                    dfs(neighbor_node, visited, component)
    
    # Find all connected components
    visited = set()
    multi_view_associations = []
    
    for view in graph:
        for instance in graph[view]:
            node = (view, instance)
            if node not in visited:
                component = set()
                dfs(node, visited, component)
                multi_view_associations.append(component)
    
    return multi_view_associations


def visualize_multi_view_associations(img_paths, multi_view_associations, bboxes, save_dir='vis'):
    imgs = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        imgs.append(img)
    for k in range(len(multi_view_associations)):
        ins_set = multi_view_associations[k]
        for cam_id, idx in ins_set:
            bbox = bboxes[cam_id][idx]
            color = get_color(k)
            h, w, _ = imgs[cam_id].shape
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int(bbox[2] * w)
            y2 = int(bbox[3] * h)
            cv2.rectangle(imgs[cam_id], [x1, y1], [x2, y2], color=color, thickness=3)
            label_color = (0, 255, 0)
            cv2.putText(imgs[cam_id], str(k), [x1, y2], cv2.FONT_HERSHEY_PLAIN, 4, label_color, thickness=3)
    for i, img in enumerate(imgs):
        name = img_paths[i].rsplit('/', 1)[1].rsplit('.', 1)[0]
        cv2.imwrite(os.path.join(save_dir, name + '.jpg'), img)
