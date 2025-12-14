import torch
import numpy as np

def eval_center_depth(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return {'mse': mse.item()}


def eval_depth(pred, target):
    assert pred.shape == target.shape

    # p_mean, d_mean = torch.mean(pred), torch.mean(target)
    # k = torch.sum((pred-p_mean)*(target-d_mean))/torch.sum((pred-p_mean)**2)
    # b = d_mean - k*p_mean
    # pred = k*pred + b

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.mean((thresh < 1.25).float()) 
    d2 = torch.mean((thresh < 1.25 ** 2).float()) 
    d3 = torch.mean((thresh < 1.25 ** 3).float()) 
    epsilon=1e-3 # to avoid log(0) NaN
    diff = pred - target
    diff_log = torch.log(pred+epsilon) - torch.log(target+epsilon)

    abs_rel = torch.mean(torch.abs(diff) / (target+epsilon))
    sq_rel = torch.mean(torch.pow(diff, 2) / (target+epsilon))

    mse = torch.mean(torch.pow(diff, 2))
    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred+epsilon) - torch.log10(target+epsilon)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(), 
            'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'log10':log10.item(), 'silog':silog.item(), 'mse': mse.item()}

def eval_depth_numpy(pred, target):
    # p_mean, d_mean = np.mean(pred), np.mean(target)
    # k = np.sum((pred-p_mean)*(target-d_mean))/np.sum((pred-p_mean)**2)
    # b = d_mean - k*p_mean
    # pred = k*pred + b

    thresh = np.maximum((target / pred), (pred / target))

    d1 = np.sum(thresh < 1.25) / len(thresh)
    d2 = np.sum(thresh < 1.25 ** 2) / len(thresh)
    d3 = np.sum(thresh < 1.25 ** 3) / len(thresh)
    epsilon=1e-3 # to avoid log(0) NaN
    diff = pred - target
    diff_log = np.log(pred+epsilon) - np.log(target+epsilon)

    abs_rel = np.mean(np.abs(diff) / (target+epsilon))
    sq_rel = np.mean(np.power(diff, 2) / (target+epsilon))

    mse = np.mean(np.power(diff, 2))
    rmse = np.sqrt(np.mean(np.power(diff, 2)))
    rmse_log = np.sqrt(np.mean(np.power(diff_log , 2)))

    log10 = np.mean(np.abs(np.log10(pred+epsilon) - np.log10(target+epsilon)))
    silog = np.sqrt(np.power(diff_log, 2).mean() - 0.5 * np.power(diff_log.mean(), 2))

    return {'d1': d1, 'd2': d2, 'd3': d3, 'abs_rel': abs_rel, 'sq_rel': sq_rel, 'rmse': rmse, 'rmse_log': rmse_log, 'log10':log10, 'silog':silog, 'mse': mse}