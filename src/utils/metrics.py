import numpy as np
import torch


class ConfusionMatrix:
    """Accumulates a KxK confusion matrix on-device; ignores labels >= K."""

    def __init__(self, num_classes: int, device="cpu"):
        self.k = num_classes
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred, target = pred.flatten(), target.flatten()
        valid = (target >= 0) & (target < self.k)
        idx = target[valid] * self.k + pred[valid]
        self.mat += torch.bincount(idx, minlength=self.k ** 2).reshape(self.k, self.k)

    def compute(self):
        m = self.mat.double().cpu().numpy()
        tp = np.diag(m)
        gt, pr = m.sum(1), m.sum(0)
        union = gt + pr - tp
        iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
        return {
            "mIoU": float(np.nanmean(iou)) * 100,
            "pixel_acc": float(tp.sum() / max(m.sum(), 1)) * 100,
            "mean_acc": float(np.nanmean(np.where(gt > 0, tp / np.maximum(gt, 1), np.nan))) * 100,
            "per_class_iou": [None if np.isnan(v) else float(v) * 100 for v in iou],
            "confusion": m.astype(np.int64).tolist(),
        }
