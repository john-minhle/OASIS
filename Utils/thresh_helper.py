import torch

class ThreshController:
    def __init__(self, nclass, momentum, thresh_init=0.85):
        self.thresh_global = torch.tensor(thresh_init, device='cuda')
        self.momentum = momentum
        self.nclass = nclass

    def new_global_mask_pooling(self, pred, ignore_mask=None):
        return_dict = {}
        mask_pred = torch.argmax(pred, dim=1)  # Lấy nhãn có xác suất cao nhất
        pred_softmax = pred.softmax(dim=1)  # Chuyển thành xác suất
        pred_conf = pred_softmax.max(dim=1)[0]  # Lấy giá trị xác suất cao nhất

        unique_cls = torch.unique(mask_pred)  # Lấy các lớp có trong batch
        cls_num = len(unique_cls)
        new_global = 0.0

        for cls in unique_cls:
            cls_map = (mask_pred == cls)
            if ignore_mask is not None:
                cls_map &= (ignore_mask != 255)
            if cls_map.sum() == 0:
                cls_num -= 1
                continue
            new_global += pred_conf[cls_map].max()

        return_dict['new_global'] = new_global / cls_num if cls_num > 0 else None
        return return_dict

    def thresh_update(self, pred, ignore_mask=None, update_g=False):
        thresh = self.new_global_mask_pooling(pred, ignore_mask)
        if update_g and thresh['new_global'] is not None:
            self.thresh_global = self.momentum * self.thresh_global + (1 - self.momentum) * thresh['new_global']

    def get_thresh_global(self):
        return self.thresh_global
