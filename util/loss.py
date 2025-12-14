import torch
from torch import nn
import torch.nn.functional as F

def ssi_normalize_depth(depth):
    median = torch.median(depth)
    abs_diff = torch.abs(depth - median)  
    mean_abs_diff = torch.mean(abs_diff)
    normalized_depth = (depth - median) / mean_abs_diff
    return normalized_depth

class TrimMAELoss:
    def __init__(self, trim=0.2):
        self.trim = trim

    def __call__(self, prediction, target):
        res = (prediction - target).abs()
        sorted_res, _ = torch.sort(res.view(-1), descending=False)
        trimmed = sorted_res[: int(len(res) * (1.0 - self.trim))]
        return trimmed.sum() / len(res)

class MultiScaleDeriLoss(nn.Module):
    def __init__(self, operator='Scharr', norm=1, scales=6, trim=False, ssi=False, amp=False):
        super().__init__()
        self.name = "MultiScaleDerivativeLoss"
        self.operator = operator
        dtype = torch.float16 if amp else torch.float
        self.operators = {
            "Scharr": {
                'x': torch.tensor([[[[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]]], dtype=dtype).cuda(),
                'y': torch.tensor([[[[-3, 10, -3], [0, 0, 0], [3, 10, 3]]]], dtype=dtype).cuda(),
            },
            "Laplace": {
                'x': torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=dtype).cuda(),
                'y': torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=dtype).cuda(),
            }
        }
        self.op_x = self.operators[operator]['x']
        self.op_y = self.operators[operator]['y']
        if norm == 1:
            self.loss_function = nn.L1Loss(reduction='mean')
        elif norm == 2:
            self.loss_function = nn.MSELoss(reduction='mean')
        if trim:
            self.loss_function = TrimMAELoss()
        self.ssi = ssi
        self.scales = scales

    def gradients(self, input_tensor):
        op_x, op_y = self.op_x, self.op_y
        groups = input_tensor.shape[1]
        op_x = op_x.repeat(groups, 1, 1, 1)
        op_y = op_y.repeat(groups, 1, 1, 1)
        grad_x = F.conv2d(input_tensor, op_x, groups=groups)
        grad_y = F.conv2d(input_tensor, op_y, groups=groups)
        return grad_x, grad_y

    def forward(self, prediction, target, mask=None):
        if self.ssi:
            prediction_ = ssi_normalize_depth(prediction)
            target_ = ssi_normalize_depth(target)
        else:
            prediction_ = prediction
            target_ = target
        prediction_ = prediction_.unsqueeze(0)
        target_ = target_.unsqueeze(0)
        total_loss = 0.0
        for scale in range(self.scales):
            grad_prediction_x, grad_prediction_y = self.gradients(prediction_)
            grad_target_x, grad_target_y = self.gradients(target_)
            loss_x = self.loss_function(grad_prediction_x, grad_target_x)
            loss_y = self.loss_function(grad_prediction_y, grad_target_y)
            total_loss += torch.mean(loss_x + loss_y)
            prediction_ = F.interpolate(prediction_, scale_factor=0.5)
            target_ = F.interpolate(target_, scale_factor=0.5)
        return total_loss / self.scales

def norm(x):
    mean = torch.mean(x)
    std = torch.std(x)
    return (x - mean) / std

class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        epsilon=1e-3 # to avoid log(0) NaN
        diff_log = torch.log(target[valid_mask]+epsilon) - torch.log(pred[valid_mask]+epsilon)
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss

def grad_mask(mask):
    return mask[..., 1:, 1:] & mask[..., 1:, :-1] & mask[..., :-1, 1:]

def gradient_x(img):
    # 计算图像在x方向的梯度
    gx = img[ :, 1:, :-1] - img[:, 1:, 1:]
    return gx

def gradient_y(img):
    # 计算图像在y方向的梯度
    gy = img[:, :-1, 1:] - img[ :, 1:, 1:]
    return gy

# class DepthAwareSmoothnessLoss(nn.Module):
#     def __init__(self):
#         super(DepthAwareSmoothnessLoss, self).__init__()

#     def forward(self, predicted_depth, true_depth, mask=None):
#         mask_g = grad_mask(mask)
        
#         # 计算预测深度图的梯度
#         pred_depth_grad_x = gradient_x(predicted_depth)
#         pred_depth_grad_y = gradient_y(predicted_depth)

#         # 计算真实深度图的梯度
#         true_depth_grad_x = gradient_x(true_depth)
#         true_depth_grad_y = gradient_y(true_depth)

#         # 计算权重：真实深度图梯度的指数衰减
#         weights_x = torch.exp(-torch.mean(torch.abs(true_depth_grad_x), 1, keepdim=True))
#         weights_y = torch.exp(-torch.mean(torch.abs(true_depth_grad_y), 1, keepdim=True))

#         # 计算加权梯度
#         smoothness_x = pred_depth_grad_x * weights_x
#         smoothness_y = pred_depth_grad_y * weights_y

#         # 计算最终的光滑损失
#         loss_x = torch.mean(torch.abs(smoothness_x[mask_g]))
#         loss_y = torch.mean(torch.abs(smoothness_y[mask_g]))

#         return loss_x + loss_y

class DepthAwareSmoothnessLoss(nn.Module):
    def __init__(self):
        super(DepthAwareSmoothnessLoss, self).__init__()

    def forward(self, pred, target, valid_mask=None):
        valid_mask = valid_mask.detach()
        pred_norm = norm(pred)
        target_norm = norm(target)
        diff_norm = pred_norm - target_norm
        diff_grad_x = torch.abs(diff_norm[..., 1:, :] - diff_norm[..., :-1, :])
        diff_grad_y = torch.abs(diff_norm[..., :, 1:] - diff_norm[..., :, :-1])
        loss = torch.mean(diff_grad_x) + torch.mean(diff_grad_y)

        return loss
    
class FeatureCosineLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(FeatureCosineLoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, feat_img, feat_event):
        loss = 0.0
        for fi, fe in zip(feat_img, feat_event):
            fi = fi.reshape(fi.size(0), -1)
            fe = fe.reshape(fe.size(0), -1)
            # print('fi',fi.shape, 'fe', fe.shape)
            fi = F.normalize(fi, p=2, dim=1)
            fe = F.normalize(fe, p=2, dim=1)
            # print('fi-fe', torch.max(fi-fe), torch.min(fi-fe))
            # print('fi',fi.shape, 'fe', fe.shape)
            valid_mask = torch.abs(fi-fe) < self.alpha 
            # print('valid_mask',valid_mask.shape)
            fi = fi[valid_mask]
            fe = fe[valid_mask]
            loss += 1 - F.cosine_similarity(fi, fe, dim=0)
        return loss
    
class FeatureGradLoss(nn.Module):
    def __init__(self):
        super(FeatureGradLoss, self).__init__()
    
    def forward(self, feat_img, feat_event):
        # loss_x, loss_y = 0.0, 0.0
        loss = 0.0
        for fi, fe in zip(feat_img, feat_event):
            pred_norm = norm(fi)
            target_norm = norm(fe)
            diff_norm = pred_norm - target_norm
            diff_grad_x = torch.abs(diff_norm[..., 1:, :] - diff_norm[..., :-1, :])
            diff_grad_y = torch.abs(diff_norm[..., :, 1:] - diff_norm[..., :, :-1])
            loss += torch.mean(diff_grad_x) + torch.mean(diff_grad_y)
        return loss
            # # mask_g = grad_mask(mask)
        
            # # 计算预测深度图的梯度
            # pred_depth_grad_x = gradient_x(fi)
            # pred_depth_grad_y = gradient_y(fi)

            # # 计算真实深度图的梯度
            # true_depth_grad_x = gradient_x(fe)
            # true_depth_grad_y = gradient_y(fe)

            # # 计算权重：真实深度图梯度的指数衰减
            # weights_x = torch.exp(-torch.mean(torch.abs(true_depth_grad_x), 1, keepdim=True))
            # weights_y = torch.exp(-torch.mean(torch.abs(true_depth_grad_y), 1, keepdim=True))

            # # 计算加权梯度
            # smoothness_x = pred_depth_grad_x * weights_x
            # smoothness_y = pred_depth_grad_y * weights_y

            # # 计算最终的光滑损失
            # loss_x = torch.mean(torch.abs(smoothness_x))
            # loss_y = torch.mean(torch.abs(smoothness_y))

            # return loss_x + loss_y