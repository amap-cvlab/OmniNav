# basic path
import torch
import numpy as np
from common.eval_det import calc_iou
from data.datasets.constant import CLASS_LABELS_200
from scipy.optimize import linear_sum_assignment
import numpy as np
import open3d as o3d
from tqdm import tqdm

def mask_matrix_nms(masks,
                    labels,
                    scores,
                    filter_thr=-1,
                    nms_pre=-1,
                    max_num=-1,
                    kernel='gaussian',
                    sigma=2.0,
                    mask_area=None):
    """Matrix NMS for multi-class masks.

    Args:
        masks (Tensor): Has shape (num_instances, m)
        labels (Tensor): Labels of corresponding masks,
            has shape (num_instances,).
        scores (Tensor): Mask scores of corresponding masks,
            has shape (num_instances).
        filter_thr (float): Score threshold to filter the masks
            after matrix nms. Default: -1, which means do not
            use filter_thr.
        nms_pre (int): The max number of instances to do the matrix nms.
            Default: -1, which means do not use nms_pre.
        max_num (int, optional): If there are more than max_num masks after
            matrix, only top max_num will be kept. Default: -1, which means
            do not use max_num.
        kernel (str): 'linear' or 'gaussian'.
        sigma (float): std in gaussian method.
        mask_area (Tensor): The sum of seg_masks.

    Returns:
        tuple(Tensor): Processed mask results.

            - scores (Tensor): Updated scores, has shape (n,).
            - labels (Tensor): Remained labels, has shape (n,).
            - masks (Tensor): Remained masks, has shape (n, m).
            - keep_inds (Tensor): The indices number of
                the remaining mask in the input mask, has shape (n,).
    """
    assert len(labels) == len(masks) == len(scores)
    if len(labels) == 0:
        return scores.new_zeros(0), labels.new_zeros(0), masks.new_zeros(
            0, *masks.shape[-1:]), labels.new_zeros(0)
    if mask_area is None:
        mask_area = masks.sum(1).float()
    else:
        assert len(masks) == len(mask_area)

    # sort and keep top nms_pre
    scores, sort_inds = torch.sort(scores, descending=True)

    keep_inds = sort_inds
    if nms_pre > 0 and len(sort_inds) > nms_pre:
        sort_inds = sort_inds[:nms_pre]
        keep_inds = keep_inds[:nms_pre]
        scores = scores[:nms_pre]
    masks = masks[sort_inds]
    mask_area = mask_area[sort_inds]
    labels = labels[sort_inds]

    num_masks = len(labels)
    flatten_masks = masks.reshape(num_masks, -1).float()
    # inter.
    inter_matrix = torch.mm(flatten_masks, flatten_masks.transpose(1, 0))
    expanded_mask_area = mask_area.expand(num_masks, num_masks)
    # Upper triangle iou matrix.
    iou_matrix = (inter_matrix /
                  (expanded_mask_area + expanded_mask_area.transpose(1, 0) -
                   inter_matrix)).triu(diagonal=1)
    # label_specific matrix.
    expanded_labels = labels.expand(num_masks, num_masks)
    # Upper triangle label matrix.
    label_matrix = (expanded_labels == expanded_labels.transpose(
        1, 0)).triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(num_masks,
                                           num_masks).transpose(1, 0)

    # IoU decay
    decay_iou = iou_matrix * label_matrix

    # Calculate the decay_coefficient
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou**2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou**2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError(
            f'{kernel} kernel is not supported in matrix nms!')
    # update the score.
    scores = scores * decay_coefficient

    if filter_thr > 0:
        keep = scores >= filter_thr
        keep_inds = keep_inds[keep]
        if not keep.any():
            return scores.new_zeros(0), labels.new_zeros(0), masks.new_zeros(
                0, *masks.shape[-1:]), labels.new_zeros(0)
        masks = masks[keep]
        scores = scores[keep]
        labels = labels[keep]

    # sort and keep top max_num
    scores, sort_inds = torch.sort(scores, descending=True)
    keep_inds = keep_inds[sort_inds]
    if max_num > 0 and len(sort_inds) > max_num:
        sort_inds = sort_inds[:max_num]
        keep_inds = keep_inds[:max_num]
        scores = scores[:max_num]
    masks = masks[sort_inds]
    labels = labels[sort_inds]

    return scores, labels, masks, keep_inds

def axis_aligned_bbox_overlaps_3d(bboxes1,
                                  bboxes2,
                                  mode='iou',
                                  is_aligned=False,
                                  eps=1e-6):
    """Calculate overlap between two set of axis aligned 3D bboxes. If
    ``is_aligned`` is ``False``, then calculate the overlaps between each bbox
    of bboxes1 and bboxes2, otherwise the overlaps between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 6) in <x1, y1, z1, x2, y2, z2>
            format or empty.
        bboxes2 (Tensor): shape (B, n, 6) in <x1, y1, z1, x2, y2, z2>
            format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or "giou" (generalized
            intersection over union).
        is_aligned (bool, optional): If True, then m and n must be equal.
            Defaults to False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Defaults to 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 0, 10, 10, 10],
        >>>     [10, 10, 10, 20, 20, 20],
        >>>     [32, 32, 32, 38, 40, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 0, 10, 20, 20],
        >>>     [0, 10, 10, 10, 19, 20],
        >>>     [10, 10, 10, 20, 20, 20],
        >>> ])
        >>> overlaps = axis_aligned_bbox_overlaps_3d(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )
    Example:
        >>> empty = torch.empty(0, 6)
        >>> nonempty = torch.FloatTensor([[0, 0, 0, 10, 9, 10]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes's last dimension is 6
    assert (bboxes1.size(-1) == 6 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 6 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 3] -
             bboxes1[..., 0]) * (bboxes1[..., 4] - bboxes1[..., 1]) * (
                 bboxes1[..., 5] - bboxes1[..., 2])
    area2 = (bboxes2[..., 3] -
             bboxes2[..., 0]) * (bboxes2[..., 4] - bboxes2[..., 1]) * (
                 bboxes2[..., 5] - bboxes2[..., 2])

    if is_aligned:
        lt = torch.max(bboxes1[..., :3], bboxes2[..., :3])  # [B, rows, 3]
        rb = torch.min(bboxes1[..., 3:], bboxes2[..., 3:])  # [B, rows, 3]

        wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :3], bboxes2[..., :3])
            enclosed_rb = torch.max(bboxes1[..., 3:], bboxes2[..., 3:])
    else:
        lt = torch.max(bboxes1[..., :, None, :3],
                       bboxes2[..., None, :, :3])  # [B, rows, cols, 3]
        rb = torch.min(bboxes1[..., :, None, 3:],
                       bboxes2[..., None, :, 3:])  # [B, rows, cols, 3]

        wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 3]
        overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :3],
                                    bboxes2[..., None, :, :3])
            enclosed_rb = torch.max(bboxes1[..., :, None, 3:],
                                    bboxes2[..., None, :, 3:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou']:
        return ious
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1] * enclose_wh[..., 2]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious

def convert_box_to_xyz(bbox):
    # bbox (m, 6)
    return torch.stack(
        (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
            bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
            bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
        dim=-1)

def voxel_downsample_point_cloud_and_mask(point_cloud, object_mask, voxel_size=0.02):
    """
    对点云和对应的 object_mask 进行体素化下采样。
    
    Args:
        point_cloud (np.ndarray): 点云数据，形状为 (N, 6)，每点包含 (x, y, z, r, g, b)。
        object_mask (np.ndarray): 点云对应的 mask 数据，形状为 (N, M)。
        voxel_size (float): 体素网格的大小。
    
    Returns:
        downsampled_point_cloud (np.ndarray): 下采样后的点云数据，形状为 (K, 6)。
        downsampled_object_mask (np.ndarray): 下采样后的 mask 数据，形状为 (K, M)。
    """
    # 获取点云的坐标 (x, y, z)
    coords = point_cloud[:, :3]
    
    # 将点云归一化到体素网格，计算每个点的体素索引
    voxel_indices = np.floor(coords / voxel_size).astype(np.int32)
    
    # 构建哈希表，去重体素索引
    _, unique_indices, inverse_indices = np.unique(voxel_indices, axis=0, return_index=True, return_inverse=True)
    
    # 使用 unique_indices 获取每个体素的第一个点
    downsampled_point_cloud = point_cloud[unique_indices]
    
    # 对 object_mask 进行同步下采样
    downsampled_object_mask = object_mask[unique_indices]

    return downsampled_point_cloud, downsampled_object_mask

class RepresentationManager:
    def __init__(self):
        # parameter for single query activation
        self.topk_single_frame_object = 15
        self.filter_out_object_min_points = 100
        self.min_object_score = 0.4
        self.kernel = 'linear'
        # parameter for query merge
        self.match_cost_min = 0.05
        self.set_class_to_zero = True
        # parameter for global activation
        self.topk_objects = 400 
        # point cloud
        self.point_cloud = np.zeros((0, 6)) # Nx6
        # query information
        self.object_mask = np.zeros((0,0)) # NxM
        self.object_class = np.zeros((0)) # M
        self.object_score = np.zeros((0)) # M
        self.object_box = np.zeros((0, 6)) # Mx6
        self.object_count = np.zeros((0)) # M
        self.object_feat = np.zeros((0, 768)) # Mx768
        self.open_vocab_feat = np.zeros((0, 768)) # Mx768
    
    def reset(self):
        self.point_cloud = np.zeros((0, 6))
        self.object_mask = np.zeros((0, 0))
        self.object_class = np.zeros((0))
        self.object_score = np.zeros((0))
        self.object_box = np.zeros((0, 6))
        self.object_count = np.zeros((0))
        self.object_feat = np.zeros((0, 768))
        self.open_vocab_feat = np.zeros((0, 768))
    
    def save_colored_point_cloud(self):
        # Create a color map for the object masks
        num_objects = self.object_mask.shape[1]
        colors = np.random.rand(num_objects, 3)  # Random colors for each object

        # Initialize the colored point cloud
        colored_point_cloud = np.ones((self.point_cloud.shape[0], 9))  # Nx6
        colored_point_cloud[:, :6] = self.point_cloud[:, :6] # Copy the point cloud data
        colored_point_cloud[:, 3:6] /= 255.0  # Normalize the RGB values to [0, 1]

        # Apply colors to the point cloud based on the object mask
        for i in range(num_objects):
            mask = self.object_mask[:, i].astype(bool)
            colored_point_cloud[mask, 6:] = colors[i]  # Assign color to the points

        # Save the colored point cloud to a .npy file
        np.save('colored_point_cloud.npy', colored_point_cloud)
        
    def merge(self, pred_dict_list):
        for idx in range(len(pred_dict_list)):
            # load data
            data = pred_dict_list[idx]
            cur_point_cloud = data['point_cloud']
            cur_mask = data['pred_masks']
            cur_class = data['pred_classes']
            if self.set_class_to_zero:
                cur_class[:] = 0
            cur_score = data['pred_scores']
            cur_mask_scores = data['pred_mask_scores']
            cur_box = data['pred_boxes']
            cur_feat = data['pred_feats']
            cur_open_vocab_feat = data['open_vocab_feats']
            # process prev data
            self.point_cloud = np.concatenate((self.point_cloud, cur_point_cloud), axis=0)
            self.object_mask = np.concatenate((self.object_mask, np.zeros((cur_point_cloud.shape[0], self.object_mask.shape[1]))), axis=0)
            # cur query activation
            if cur_mask is None:
                continue
            else:
                # sort according to score
                sorted_indices = np.argsort(-cur_score)
                cur_score = cur_score[sorted_indices]
                cur_mask = cur_mask[:, sorted_indices]
                cur_class = cur_class[sorted_indices]
                cur_box = cur_box[sorted_indices]
                cur_mask_scores = cur_mask_scores[sorted_indices]
                cur_feat = cur_feat[sorted_indices]
                cur_open_vocab_feat = cur_open_vocab_feat[sorted_indices]
                # token topk_single_frame_object
                if cur_mask.shape[1] > self.topk_single_frame_object:
                    cur_mask = cur_mask[:, :self.topk_single_frame_object]
                    cur_class = cur_class[:self.topk_single_frame_object]
                    cur_score = cur_score[:self.topk_single_frame_object]
                    cur_box = cur_box[:self.topk_single_frame_object, :]
                    cur_mask_scores = cur_mask_scores[:self.topk_single_frame_object]
                    cur_feat = cur_feat[:self.topk_single_frame_object]
                    cur_open_vocab_feat = cur_open_vocab_feat[:self.topk_single_frame_object]
                # normalize score
                cur_score = cur_score * cur_mask_scores
                # nms
                cur_score, cur_class, cur_mask, keep_inds = mask_matrix_nms(torch.tensor(cur_mask).float().transpose(0, 1), torch.tensor(cur_class).long(), torch.tensor(cur_score).float(), kernel=self.kernel)
                cur_score = cur_score.numpy()
                cur_class = cur_class.numpy()
                cur_mask = cur_mask.transpose(0, 1).numpy()
                cur_box = cur_box[keep_inds, :]
                cur_feat = cur_feat[keep_inds, :]
                cur_open_vocab_feat = cur_open_vocab_feat[keep_inds, :]
                if cur_box.ndim == 1:
                    cur_box = cur_box.reshape(1, -1)
                if cur_feat.ndim == 1:
                    cur_feat = cur_feat.reshape(1, -1)
                if cur_open_vocab_feat.ndim == 1:
                    cur_open_vocab_feat = cur_open_vocab_feat.reshape(1, -1)
                # no object continue
                if cur_mask.shape[1] == 0:
                    continue
                # score thr
                score_mask = cur_score > self.min_object_score
                cur_score = cur_score[score_mask]
                cur_class = cur_class[score_mask]
                cur_mask = cur_mask[:, score_mask]
                cur_box = cur_box[score_mask]
                cur_feat = cur_feat[score_mask]
                cur_open_vocab_feat = cur_open_vocab_feat[score_mask]
                # no object continue
                if cur_mask.shape[1] == 0:
                    continue
                # num points thr
                num_points_mask = np.sum(cur_mask, axis=0) > self.filter_out_object_min_points
                cur_score = cur_score[num_points_mask]
                cur_class = cur_class[num_points_mask]
                cur_mask = cur_mask[:, num_points_mask]
                cur_box = cur_box[num_points_mask]
                cur_feat = cur_feat[num_points_mask]
                cur_open_vocab_feat = cur_open_vocab_feat[num_points_mask]
                # no object continue
                if cur_mask.shape[1] == 0:
                    continue
            # process cur data
            cur_mask = np.concatenate((np.zeros((self.object_mask.shape[0] - cur_mask.shape[0], cur_mask.shape[1])), cur_mask), axis=0)
            # query merging
            # get merge id
            box_iou_cost = axis_aligned_bbox_overlaps_3d(convert_box_to_xyz(torch.Tensor(self.object_box).unsqueeze(0)), convert_box_to_xyz(torch.Tensor(cur_box).unsqueeze(0)), mode='iou', is_aligned=False).squeeze(0)
            class_cost = torch.where(torch.Tensor(self.object_class).unsqueeze(1) == torch.Tensor(cur_class).unsqueeze(0), torch.ones_like(box_iou_cost), torch.zeros_like(box_iou_cost))
            # open_vocab_cost = torch.mm(
            #     torch.nn.functional.normalize(torch.Tensor(self.open_vocab_feat), p=2, dim=1),
            #     torch.nn.functional.normalize(torch.Tensor(cur_open_vocab_feat), p=2, dim=1).transpose(0, 1)
            # )
            mix_cost = box_iou_cost * class_cost
            row_ind, col_ind = linear_sum_assignment(-mix_cost.cpu())
            mix_cost_mask = mix_cost[row_ind, col_ind] > self.match_cost_min
            row_ind = (torch.Tensor(row_ind).int()[mix_cost_mask]).numpy()
            col_ind = (torch.Tensor(col_ind).int()[mix_cost_mask]).numpy()
            # merging
            self.object_mask[:, row_ind] = np.logical_or(self.object_mask[:, row_ind], cur_mask[:, col_ind])
            self.object_score[row_ind] = self.object_score[row_ind] * (self.object_count[row_ind]) / (self.object_count[row_ind] + 1) + cur_score[col_ind] / (self.object_count[row_ind] + 1)
            self.object_box[row_ind] = self.object_box[row_ind] * np.expand_dims(self.object_count[row_ind], axis=1) / np.expand_dims((self.object_count[row_ind] + 1), axis=1) + cur_box[col_ind] / np.expand_dims((self.object_count[row_ind] + 1), axis=1)
            self.object_feat[row_ind] = self.object_feat[row_ind] * np.expand_dims(self.object_count[row_ind], axis=1) / np.expand_dims((self.object_count[row_ind] + 1), axis=1) + cur_feat[col_ind] / np.expand_dims((self.object_count[row_ind] + 1), axis=1)
            self.open_vocab_feat[row_ind] = self.open_vocab_feat[row_ind] * np.expand_dims(self.object_count[row_ind], axis=1) / np.expand_dims((self.object_count[row_ind] + 1), axis=1) + cur_open_vocab_feat[col_ind] / np.expand_dims((self.object_count[row_ind] + 1), axis=1)
            self.object_count[row_ind] += 1
            # add new mask
            new_query_ind = np.ones((cur_mask.shape[1]), dtype=bool)
            new_query_ind[col_ind] = False
            self.object_mask = np.concatenate((self.object_mask, cur_mask[:, new_query_ind]), axis=1)
            self.object_class = np.concatenate((self.object_class, cur_class[new_query_ind]), axis=0)
            self.object_score = np.concatenate((self.object_score, cur_score[new_query_ind]), axis=0)
            self.object_box = np.concatenate((self.object_box, cur_box[new_query_ind]), axis=0)
            self.object_feat = np.concatenate((self.object_feat, cur_feat[new_query_ind]), axis=0)
            self.open_vocab_feat = np.concatenate((self.open_vocab_feat, cur_open_vocab_feat[new_query_ind]), axis=0)
            self.object_count = np.concatenate((self.object_count, np.ones((np.sum(new_query_ind)))), axis=0)
            # global activation
            # If object_mask has more objects than topk_objects, select topk_objects based on top object_score
            if self.object_mask.shape[1] > self.topk_objects:
                topk_indices = np.argsort(self.object_score)[-self.topk_objects:]
                self.object_mask = self.object_mask[:, topk_indices]
                self.object_class = self.object_class[topk_indices]
                self.object_score = self.object_score[topk_indices]
                self.object_box = self.object_box[topk_indices]
                self.object_feat = self.object_feat[topk_indices]
                self.open_vocab_feat = self.open_vocab_feat[topk_indices]
                self.object_count = self.object_count[topk_indices]
        self.point_cloud, self.object_mask = voxel_downsample_point_cloud_and_mask(self.point_cloud, self.object_mask)