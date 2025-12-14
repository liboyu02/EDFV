import torch
import torch.nn as nn

from util.utils import init_log
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import logging
import torch.nn.functional as F
from util.train_utils import fix_random_seed, register_nan_detection_hook


from dataset.fullnet_data import MyFullNetDataset
from util.metric import eval_depth

from model.fullnet import FullNet
from util.loss import *
# from model.conv import DepthPredictionCNN
import glob
from util.local_ssi import EdgeGuidedLocalSSI


train_paths = sorted(glob.glob('/mnt/E/EvFaster/data/fix30_100fl/train/*'))
val_paths = sorted(glob.glob('/mnt/E/EvFaster/data/fix30_100fl/test/*'))

bins=32
min_depth=0.1
max_depth=30.0

def get_dataloaders(batch_size):
    
    train_dataset = MyFullNetDataset(train_paths, mode='train',bins = bins,min_depth=min_depth, max_depth=max_depth, norm=False,size=(320,640), randmask=True)
    
    val_dataset = MyFullNetDataset(val_paths, mode='val',bins = bins, min_depth=min_depth, max_depth=max_depth, norm=False,size=(320,640))
    
           
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                  batch_size = batch_size,
                                                  shuffle=True,
                                                  num_workers=4,
                                                  drop_last=True
                                                  )

    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                               batch_size = 1,
                                               shuffle=False,
                                               num_workers=4,
                                               drop_last=True
                                                )
    
    return train_dataloader, val_dataloader



def train_fn(seed, model, lr,weight_decay,batch_size,num_epochs,save_path,lamda=10.0, checkpoint_path=None):
    # print('model',model.device)
    fix_random_seed(seed)
    os.makedirs(save_path, exist_ok=True)
    register_nan_detection_hook(model)

    writer = SummaryWriter(save_path)
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    
    optim = torch.optim.AdamW(model.parameters(),lr=lr, weight_decay=weight_decay)
    
    criterion = nn.MSELoss().cuda()
    
    edgessiloss = EdgeGuidedLocalSSI(weight=1.0)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optimizer'])
        
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0

    train_dataloader, val_dataloader = get_dataloaders(batch_size)
        
    # best_val_abs_rel = 1000
    best_d1 = 0
    best_rmse = 1000

    for epoch in range(start_epoch,num_epochs):
        
        model.train()
        train_loss = 0
        tot_final_loss = 0
        tot_sparse_loss = 0
        tot_smooth_loss = 0
        tot_medium_loss = [0,0,0,0]
        
        logger.info(f"epoch_{epoch}, training...")

        
        for mydict in tqdm(train_dataloader):
            optim.zero_grad()
            depth, image_patch = mydict['depth'], mydict['image']
            voxel = mydict['event_stack'].float().cuda()
            da_depth = mydict['da_depth'].float().cuda()
            min_depth = mydict['min_depth']
            max_depth = mydict['max_depth']
            valid_mask = mydict['valid_mask'].cuda()
            
            
            depth = depth.float().cuda()

            image_patch = image_patch.float().cuda()
            
            pred, sparse, outputs = model(voxel, da_depth, valid_mask)
            
            depth_vals = np.linspace(min_depth, max_depth, bins, endpoint=True)
            depth_vals = torch.tensor(depth_vals).cuda().float()
            depth_vals = depth_vals.permute(1,0).unsqueeze(2).unsqueeze(3)#.expand_as(pred)

            
            
            pred = torch.sum(pred*depth_vals, dim=1)
            sparse = torch.sum(sparse*depth_vals, dim=1)
            
            final_loss = criterion(pred.squeeze(), depth.squeeze())
            
            smooth_loss = edgessiloss(depth.unsqueeze(1), pred.unsqueeze(1), mask=torch.ones_like(depth.unsqueeze(1)),image=image_patch).sum()
            #
            tot_smooth_loss += smooth_loss.detach()
            loss = final_loss
            if valid_mask.sum() > 0:
                sparse_loss =  lamda*criterion(sparse[valid_mask], depth[valid_mask])
                
                loss = loss + sparse_loss
                tot_sparse_loss += sparse_loss
            
            loss = loss + smooth_loss*5 # 5 blender 1 sintel
            
            smooth_loss_mediums = []
            for i in range(len(outputs)):
                depth_resize = F.interpolate(depth.unsqueeze(1), size=outputs[i].shape[-2:], mode='bilinear', align_corners=True).squeeze()
                
                image_patch_resize = F.interpolate(image_patch, size=outputs[i].shape[-2:], mode='bilinear', align_corners=True)
                smooth_loss_medium = edgessiloss(depth_resize.unsqueeze(1), torch.sum(outputs[i],axis=1,keepdim=True), mask=torch.ones_like(depth_resize.unsqueeze(1)),image=image_patch_resize).sum()
                smooth_loss_mediums.append(smooth_loss_medium.detach())
                loss = loss + smooth_loss_medium*0.1 # 0.1 blender 0.1 sintel

            for i in range(len(outputs)):
                tot_medium_loss[i] += smooth_loss_mediums[i]
            
            loss.backward()
            
            optim.step()
            
            
            train_loss += loss.detach()
            tot_final_loss += final_loss.detach()
            
                    
            
        train_loss /= len(train_dataloader)
        train_loss = train_loss.item()

        tot_final_loss /= len(train_dataloader)
        tot_sparse_loss /= len(train_dataloader)
        tot_smooth_loss /= len(train_dataloader)
        for i in range(len(outputs)):
            tot_medium_loss[i] /= len(train_dataloader)
            writer.add_scalar(f'train/medium_loss_{i}', tot_medium_loss[i], epoch)
        
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/final_loss', tot_final_loss, epoch)
        writer.add_scalar('train/sparse_loss', tot_sparse_loss, epoch)
        writer.add_scalar('train/smooth_loss', tot_smooth_loss, epoch)
        print(f"epoch_{epoch},  train_loss = {train_loss:.5f}, final_loss = {tot_final_loss:.5f}, sparse_loss = {tot_sparse_loss:.5f}, smooth_loss = {tot_smooth_loss:.5f}, medium_loss = {tot_medium_loss}")
        
        vis_pred = (pred-pred.min())/(pred.max()-pred.min())
        vis_depth = (depth-depth.min())/(depth.max()-depth.min())
        vis_sparse = (sparse-sparse.min())/(sparse.max()-sparse.min())

        writer.add_image('train/sparse', vis_sparse[0].unsqueeze(0), epoch)
        writer.add_image('train/pred', vis_pred[0].unsqueeze(0), epoch)
        writer.add_image('train/depth', vis_depth[0].unsqueeze(0), epoch)
        writer.add_scalar('train/lr', optim.param_groups[0]['lr'], epoch)

       
        # visualize some samples
        vis_img = image_patch[0]
        
        writer.add_image('train/image', vis_img, epoch)
        
        model.eval()
        results = {'d1': 0,'d2':0,'d3':0, 'abs_rel': 0,'sq_rel':0,'rmse': 0, 'rmse_log':0,'log10':0, 'silog': 0}
        logger.info(f"epoch_{epoch}, validating...")
        for mydict in tqdm(val_dataloader):
            depth, image_patch = mydict['depth'], mydict['image']
            da_depth = mydict['da_depth'].float().cuda()
            voxel = mydict['event_stack'].float().cuda()
            min_depth = mydict['min_depth']
            max_depth = mydict['max_depth']
            valid_mask = mydict['valid_mask'].cuda()
            depth_vals = np.linspace(min_depth, max_depth, bins, endpoint=True)
            depth_vals = torch.tensor(depth_vals).cuda().float()
            depth = depth.float().cuda()
            image_patch = image_patch.float().cuda()

            with torch.no_grad():
                
                pred, sparse, outputs = model(voxel, da_depth, valid_mask)
                
               
            
            pred = torch.clamp(pred, 0 ,1)
            depth_vals = depth_vals.permute(1,0).unsqueeze(2).unsqueeze(3).expand_as(pred)
            pred = torch.sum(pred*depth_vals, dim=1)
            h,w = depth.shape[-2:]
            pred = F.interpolate(pred[:,None], (h,w), mode="bilinear", align_corners=True)
            # print('pred',pred.shape, 'depth',depth.shape)
            cur_results = eval_depth(pred.squeeze().cpu()/max_depth, depth.squeeze().cpu()/max_depth)
            
            for k in results.keys():
                results[k] += cur_results[k]
        for k in results.keys():
            results[k] = results[k] / len(val_dataloader)
            
        
        for name, metric in results.items():
                writer.add_scalar(f'eval/{name}', metric, epoch)            
        
        checkpoint = {
                'model': model.state_dict(),
                'optimizer': optim.state_dict(),
                'epoch': epoch
            }
        torch.save(checkpoint, os.path.join(save_path, 'latest.pth'))

        
        if results['d1'] > best_d1:
            best_d1 = results['d1']
            logger.info(f"epoch_{epoch}, saving best model with d1 = {best_d1:.5f}")
            torch.save(checkpoint, os.path.join(save_path, f'best_d1.pth'))
        if results['rmse'] < best_rmse:
            best_rmse = results['rmse']
            logger.info(f"epoch_{epoch}, saving best model with rmse = {best_rmse:.5f}")
            torch.save(checkpoint, os.path.join(save_path, f'best_rmse.pth'))

        logger.info(f"epoch_{epoch},  train_loss = {train_loss:.5f}, val_metrics = {results}")
        


model = FullNet(in_channel_event=bins, in_channel_image=1, out_channels=bins, mid_channels=8, sparse_channels=bins)
model.cuda()

train_fn(42, model, lr=3e-4,weight_decay=1e-5,batch_size=4,num_epochs=700,save_path='checkpoint/nips25_fullnet/',lamda=0.1, checkpoint_path=None)
