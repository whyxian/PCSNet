from sklearn.metrics import roc_auc_score, auc
import numpy as np
from skimage.measure import label, regionprops
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F


def roc_auc_img(gt, score):
    img_roc_auc = roc_auc_score(gt, score)

    return img_roc_auc


def roc_auc_pxl(gt, score):
    if gt.sum() == 0 or gt.sum() == gt.size:
        return 0.0
    per_pixel_roc_auc = roc_auc_score(gt.flatten(), score.flatten())
    return per_pixel_roc_auc


def pro_auc_pxl(gt, score):
    gt = np.squeeze(gt, axis=1)

    gt[gt <= 0.5] = 0
    gt[gt > 0.5] = 1
    gt = gt.astype(np.bool)

    if gt.sum() == 0:  # 没有异常像素，返回 0
        return 0.0

    max_step = 200
    expect_fpr = 0.3

    max_th = score.max()
    min_th = score.min()
    delta = (max_th - min_th) / max_step

    pros_mean = []
    fprs = []

    binary_score_maps = np.zeros_like(score, dtype=np.bool)

    for step in range(max_step):
        thred = max_th - step * delta
        binary_score_maps[score <= thred] = 0
        binary_score_maps[score > thred] = 1

        pro = []
        for i in range(len(binary_score_maps)):
            label_map = label(gt[i], connectivity=2)
            props = regionprops(label_map, binary_score_maps[i])

            for prop in props:
                pro.append(prop.intensity_image.sum() / prop.area)

        if len(pro) > 0:
            pros_mean.append(np.array(pro).mean())
        else:
            pros_mean.append(0.0)

        gt_neg = ~gt
        fpr = np.logical_and(gt_neg, binary_score_maps).sum() / gt_neg.sum()
        fprs.append(fpr)

    pros_mean = np.array(pros_mean)
    fprs = np.array(fprs)

    idx = fprs <= expect_fpr
    if idx.sum() == 0:
        return 0.0
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)
    pros_mean_selected = rescale(pros_mean[idx])
    per_pixel_roc_auc = auc(fprs_selected, pros_mean_selected)

    return per_pixel_roc_auc


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


def get_threshold(gt, score):
    gt_mask = np.asarray(gt)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), score.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]

    return threshold


def gaussian_smooth(x, sigma=4):
    bs = x.shape[0]
    for i in range(0, bs):
        x[i] = gaussian_filter(x[i], sigma=sigma)

    return x


def upsample(x, size, mode):
    return F.interpolate(x.unsqueeze(1), size=size, mode=mode, align_corners=False).squeeze().numpy()


def cal_img_roc(scores, gt_list):
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list)
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_img(gt_list, img_scores)

    return fpr, tpr, img_roc_auc


def cal_pxl_roc(gt_mask, scores):
    gt_flat = gt_mask.flatten()
    score_flat = scores.flatten()
    if gt_flat.sum() == 0 or gt_flat.sum() == len(gt_flat):
        return None, None, 0.0
    fpr, tpr, _ = roc_curve(gt_flat, score_flat)
    per_pixel_rocauc = roc_auc_pxl(gt_flat, score_flat)

    return fpr, tpr, per_pixel_rocauc


def cal_pxl_pro(gt_mask, scores):
    per_pixel_proauc = pro_auc_pxl(gt_mask, scores)

    return per_pixel_proauc
