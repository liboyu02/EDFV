import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from util.metric import eval_depth

import matplotlib

from model.fullnet import FullNet
import cv2
from dataset.fullnet_data import MyFullNetDataset
import glob
import time
import sys

da = sys.argv[1]

sel = 'upload'
ckpt_path = f'checkpoint/nips25/'



val_paths = sorted(glob.glob('/mnt/E/EvFaster/data/fix30_100fl/test/*/'))

save_path = f'finetune_results/nips25/scratch_{da}_{ckpt_path.split("/")[-1]}_{sel}/'
os.makedirs(save_path, exist_ok=True)
bins = 32

# model = MyResNet(in_channels=5, output_channels=1)
# model = FullNet(in_channel_event=1, in_channel_image=1, out_channels=1, mid_channels=8)
model = FullNet(in_channel_event=bins, in_channel_image=1, out_channels=bins, mid_channels=8, sparse_channels=bins)
model.load_state_dict(torch.load(f'{ckpt_path}/{sel}.pth', map_location='cpu')['model'])
model.cuda().eval()
print(val_paths)
val_dataset = MyFullNetDataset(val_paths, mode='val',bins = bins, min_depth=3.1, max_depth=30, norm=False,size=(320,640),da_result=da)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
results = {'d1': 0,'d2':0,'d3':0, 'abs_rel': 0,'sq_rel':0,'rmse': 0, 'rmse_log':0,'log10':0, 'silog': 0, 'mse': 0}
cmap = matplotlib.colormaps.get_cmap('Spectral_r')

def colorize_cv2(depth):
       depth = (depth - depth.min()) / (depth.max() - depth.min())
       vis_depth = cv2.applyColorMap((depth.squeeze().cpu().numpy()*255).astype(np.uint8), cv2.COLORMAP_JET)
       return vis_depth.astype(np.float64)/255.0
def colorize_max_depth(depth):
   
       depth = (depth-min_depth) / (max_depth-min_depth)
       vis_depth = cv2.applyColorMap((depth.squeeze().cpu().numpy()*255).astype(np.uint8), cv2.COLORMAP_JET)
       return vis_depth.astype(np.float64)/255.0
idx=0
file_f = open(f'{save_path}/results.txt','w')
pred_dir = f'{save_path}/npy/'
os.makedirs(pred_dir, exist_ok=True)
tot_time = 0

for sample in tqdm(val_dataloader):
            
            img, depth,voxel = sample['image'].cuda().float(), sample['depth'][0].cuda().float(), sample['event_stack'].cuda().float()
            da_depth = sample['da_depth'].cuda().float()
            min_depth = sample['min_depth']
            max_depth = sample['max_depth']

            valid_mask = sample['valid_mask'].cuda()
            # evaluate on the original resolution 
            depth_vals = np.linspace(min_depth, max_depth, bins, endpoint=True)
            depth_vals = torch.tensor(depth_vals).cuda().float()
            depth_vals = depth_vals.unsqueeze(0).unsqueeze(2)#.unsqueeze(3)#.expand_as(pred)
            
            with torch.no_grad():
            
                start_time = time.time()
                pred, sparse, outputs = model(voxel, da_depth, valid_mask)
                # print('pred',pred.shape, 'depth_vals',depth_vals.shape)
                pred = torch.sum(pred*depth_vals, dim=1)
                end_time = time.time()
                tot_time += end_time - start_time
                
                pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)
                
                
                sparse = torch.sum(sparse*depth_vals, dim=1)
                
            
            np.save(f'{pred_dir}/{idx:06d}.npy',pred.cpu().numpy().squeeze())
            vis_pred = colorize_max_depth(pred.cpu())
            # print('depth',depth.shape)
            vis_depth = colorize_max_depth(depth.cpu())
            
            vis_sparse = colorize_max_depth(sparse.cpu())
            vis_sparse = cv2.resize(vis_sparse, (depth.shape[-1], depth.shape[-2]), interpolation=cv2.INTER_NEAREST)
            vis_img = F.interpolate(img, depth.shape[-2:], mode='bilinear', align_corners=True)
            vis_img = vis_img[0].permute(1,2,0).cpu().numpy().astype(np.float64)
            vis_da = colorize_cv2(da_depth.cpu())
            valid_mask = cv2.resize(valid_mask.permute(1,2,0).cpu().numpy(), (depth.shape[-1], depth.shape[-2]), interpolation=cv2.INTER_NEAREST)
            valid_mask = np.expand_dims(valid_mask, axis=2).repeat(3, axis=2).astype(np.float64)
            vis_da = cv2.resize(vis_da, (vis_pred.shape[-2], vis_pred.shape[-3]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f'{save_path}/{idx}_concat.png', cv2.hconcat([vis_pred*255, vis_depth*255, vis_img*255, vis_sparse*255, valid_mask*255, vis_da*255]))

            pred = torch.clamp(pred.cpu(),min_depth,max_depth)
            cur_results = eval_depth(pred.squeeze()/max_depth, depth.cpu().squeeze()/max_depth)
            print(f"{idx} : {cur_results}",file=file_f)
            
            idx+=1
            for k in results.keys():
                results[k] += cur_results[k]
for k in results.keys():
            results[k] = results[k] / len(val_dataloader)
# 将结果写入文件
print(f"Results: {results}\nmodel_path: {ckpt_path}\nval_paths: {val_paths}",file=file_f)
print(f"average time: {tot_time/len(val_dataloader)}",file=file_f)
print(f"Results: {results}")
print(f"average time: {tot_time/len(val_dataloader)}")
file_f.close()