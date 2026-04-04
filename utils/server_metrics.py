import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype("int") + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        return count.reshape(self.num_class, self.num_class)

    def _safe_divide(self, numerator, denominator):
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float64),
            where=denominator != 0,
        )

    def calculate_iou(self):
        intersection = np.diag(self.confusion_matrix)
        union = (
            np.sum(self.confusion_matrix, axis=1)
            + np.sum(self.confusion_matrix, axis=0)
            - intersection
        )
        iou = self._safe_divide(intersection, union)
        dice = self._safe_divide(
            2 * intersection,
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0),
        )
        recall = self._safe_divide(intersection, np.sum(self.confusion_matrix, axis=1))
        return float(iou[1]), float(dice[1]), float(recall[1])

    def Pixel_Accuracy(self):
        if self.confusion_matrix.sum() == 0:
            return 0.0
        return float(np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum())

    def Pixel_Accuracy_Class(self):
        acc = self._safe_divide(
            np.diag(self.confusion_matrix),
            self.confusion_matrix.sum(axis=1),
        )
        return float(np.nanmean(acc))

    def Mean_Intersection_over_Union(self):
        miou = self._safe_divide(
            np.diag(self.confusion_matrix),
            (
                np.sum(self.confusion_matrix, axis=1)
                + np.sum(self.confusion_matrix, axis=0)
                - np.diag(self.confusion_matrix)
            ),
        )
        return float(np.nanmean(miou))

    def compute_metrics(self):
        foreground_iou, foreground_dice, foreground_recall = self.calculate_iou()
        return {
            "IoU": foreground_iou,
            "Dice": foreground_dice,
            "Recall": foreground_recall,
            "mIoU": self.Mean_Intersection_over_Union(),
            "mACC": self.Pixel_Accuracy_Class(),
        }

    def add_batch(self, gt_image, pre_image):
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
