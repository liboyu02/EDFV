import cv2
import torch
from torch.utils.data import Dataset
from glob import glob
import numpy as np
import albumentations as A

# from .util.blocks import OutlierRemoval
import random
from scipy.signal import convolve2d
from event_process import *



class MyFullNetDataset(Dataset):
    def __init__(self, root_paths, mode,  min_depth=1.0, max_depth=20,  dataset='blender', bins=64, norm=False, size=(320,640), train_crop=True, da_result='da',randmask=False):
       
        
        self.mode = mode
        # self.size = size
        self.dataset = dataset
        # with open(filelist_path, 'r') as f:
        #     self.filelist = f.read().splitlines()
        self.img_paths = []
        self.depth_paths = []
        self.event_paths = []
        # self.evgrad_paths = []
        self.sparse_depth_paths = []
        self.mask_paths = []
        self.cnts_paths = []
        self.voxel_paths = []
        self.da_paths = []
        self.foc_paths = []
        self.valid_mask_paths = []
        self.grad_paths = []
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.train_crop = train_crop
        self.randmask = randmask
        for root_path in root_paths:
            if dataset == 'blender':
                self.img_paths += glob(root_path + '/allInFocus/*.png')
                self.depth_paths += glob(root_path + '/depth/*.npy')
                if da_result == 'da':
                    self.da_paths += glob(root_path + '/DAdepthnpy_vitl/*.npy')
                elif da_result == 'scratchda':
                    self.da_paths += glob(root_path + '/DAdepthnpy_vitl_scratch/*.npy')
                elif da_result == 'metric3d':
                    self.da_paths += glob(root_path + '/Metric3D_vits/*.npy')
                elif da_result == 'depth-pro':
                    self.da_paths += glob(root_path + '/depth-pro/*.npy')
                elif da_result == 'unidepth':
                    self.da_paths += glob(root_path + '/UniDepth/*.npy')
                self.event_paths += glob(root_path + '/v2e_output/*/output.txt')
                self.foc_paths += glob(root_path + '/focus_positions/*.txt')
                self.cnts_paths += glob(root_path + '/cnts/*.npy')
                self.voxel_paths += glob(root_path + '/voxel_comp_32/*.npy')
                self.valid_mask_paths += glob(root_path + '/mask_0.1_300_dilate/*.npy')
                self.grad_paths += glob(root_path + '/grad/*.npy')
               
            elif dataset == 'sintel':
                self.img_paths += glob(root_path + f'/clean/*.png',recursive=True)
                
                self.depth_paths += glob(root_path + f'/depth_npy/*.npy',recursive=True)
                if da_result == 'da':
                    self.da_paths += glob(root_path + '/DAdepthnpy_vitl/*.npy')
                elif da_result == 'metric3d':
                    self.da_paths += glob(root_path + '/Metric3D_vitl/*.npy')
                elif da_result == 'depth-pro':
                    self.da_paths += glob(root_path + '/depth-pro/*.npy')
                elif da_result == 'unidepth':
                    self.da_paths += glob(root_path + '/UniDepth/*.npy')
                self.event_paths += glob(root_path + '/v2e_output/*/output.txt',recursive=True)
                
                self.mask_paths += glob(root_path + '/invalid/*.png',recursive=True)
                self.foc_paths += glob(root_path + '/focus_positions/*.txt')
                self.cnts_paths += glob(root_path + '/cnts/*.npy')
                self.voxel_paths += glob(root_path + '/voxel_comp_32/*.npy')
                self.valid_mask_paths += glob(root_path + '/mask_0.1_300_dilate/*.npy')
                self.grad_paths += glob(root_path + '/grad/*.npy')
                
            elif dataset == '4DLFD':
                self.img_paths += glob(root_path + '/allInFocus/*.png')
                self.depth_paths += glob(root_path + '/depth/*.npy')
                self.da_paths += glob(root_path + '/DAdepthnpy_vitl/*.npy')
                self.event_paths += glob(root_path + '/event_split_reverse_cropped/*.npy')
        
        self.img_paths.sort()
        self.depth_paths.sort()
        self.event_paths.sort()
        self.mask_paths.sort()
        self.da_paths.sort()
        self.foc_paths.sort()
        self.cnts_paths.sort()
        self.voxel_paths.sort()
        self.valid_mask_paths.sort()
        self.grad_paths.sort()
        self.da_result = da_result
        self.bins = bins
        self.norm = norm
        self.size = size

        print(f"dataset {dataset} has {len(self.img_paths)} images, {len(self.depth_paths)} depths, {len(self.event_paths)} events, {len(self.da_paths)} dense depths")
        assert len(self.img_paths) == len(self.depth_paths) == len(self.da_paths) 

        
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


    def crop(self, image, top, left, height, width):
        return image[top:top+height, left:left+width, ...]
   

    def __getitem__(self, item):
        img_path = self.img_paths[item]
        depth_path = self.depth_paths[item]
        da_path = self.da_paths[item]
        
       
        if self.dataset == 'blender' or self.dataset == 'sintel':
            
            voxel_path = self.voxel_paths[item]

            voxel = np.load(voxel_path)
            if self.bins == 16:
                voxel_grouped = voxel.reshape(voxel.shape[0], voxel.shape[1],-1,2)
                voxel = np.sum(voxel_grouped,axis=3)
            if self.bins == 8:
                voxel_grouped = voxel.reshape(voxel.shape[0], voxel.shape[1],-1,4)
                voxel = np.sum(voxel_grouped,axis=3)
            if self.bins == 4:
                voxel_grouped = voxel.reshape(voxel.shape[0], voxel.shape[1],-1,8)
                voxel = np.sum(voxel_grouped,axis=3)
            
            
            if self.mode == 'ori_test': # NO preprocessed masks
                cnts_path = self.cnts_paths[item]
                grad_path = self.grad_paths[item]
                cnts_patch = np.load(cnts_path)
                img_magnitude_norm = np.load(grad_path)
                
                candidate = np.ones_like(img_magnitude_norm, dtype=np.bool_)     
                candidate[img_magnitude_norm<0.1] = 0 
                candidate[cnts_patch<300]=0
                valid_mask = candidate
                
                valid_mask =valid_mask.astype(np.uint8)*255
                valid_mask = cv2.dilate(valid_mask, self.kernel, iterations=1)
                valid_mask = valid_mask.astype(np.bool_)

            if not self.randmask or self.mode == 'val':
                valid_mask = np.load(self.valid_mask_paths[item])
            elif self.mode == 'train' and self.randmask:
                cnts_path = self.cnts_paths[item]
                grad_path = self.grad_paths[item]
                cnts_patch = np.load(cnts_path)
                img_magnitude_norm = np.load(grad_path)
                candidate = np.ones_like(img_magnitude_norm, dtype=np.bool_)
                img_mag_thres = np.random.uniform(0., 0.2) # 0 0.2 0.09 0.11
                cnts_patch_thres = np.random.randint(0, 600) # 0 600 250 350
                
                candidate[img_magnitude_norm<img_mag_thres] = 0

                candidate[cnts_patch<cnts_patch_thres]=0

                valid_mask = candidate
                
                valid_mask =valid_mask.astype(np.uint8)*255
                valid_mask = cv2.dilate(valid_mask, self.kernel, iterations=1)
                valid_mask = valid_mask.astype(np.bool_)

        image = cv2.imread(img_path)
        image_ori = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255.0
        
        h, w = image.shape[:2]
        
            

        if self.dataset == 'blender':
            # foc_path = self.foc_paths[item]
            depth = np.load(depth_path)
            # min_depth, max_depth = 0.1, 30.0
            min_depth, max_depth = 3.1, 30.0
            depth = np.clip(depth, min_depth, max_depth)
        elif self.dataset == 'sintel':
            # depth = sio.depth_read(depth_path)
            depth = np.load(depth_path)
            depth = depth.astype(np.float32)
            
            min_depth, max_depth = 1.1, 30.0
            depth = np.clip(depth, min_depth, max_depth)
            # print('depth',depth.shape, depth.max(), depth.min())
        elif self.dataset == '4DLFD':
            depth = np.load(depth_path)
            depth = depth.astype(np.float32)
            min_depth, max_depth = -2.5,2.5
            depth = np.clip(depth, min_depth, max_depth)
        
        da_depth = np.load(da_path)
        
        if self.da_result == 'da':
            da_depth = (da_depth - da_depth.min()) / (da_depth.max() - da_depth.min())
        if self.da_result == 'metric3d':
            da_depth = (da_depth - da_depth.min()) / (da_depth.max() - da_depth.min()+1e-10)
            

        if self.dataset != 'blender' and self.dataset != 'sintel':
            print('NO preprocessed voxels!!')
            event_path = self.event_paths[item]
            img_gray = cv2.cvtColor(image_ori, cv2.COLOR_BGR2GRAY)/255.0
            if event_path.endswith('npy'):
                event_stack = np.load(event_path)
            else:
                event_stack = np.loadtxt(event_path)
            event_stack[:,0] -= event_stack[0,0]
            if self.dataset == 'blender':
                event_stack = event_stack[event_stack[:,0] >= event_stack[-1,0]/10]
            if self.dataset == 'sintel':
                event_stack = event_stack[event_stack[:,0] >= event_stack[-1,0]/30]
            elif self.dataset == '4DLFD':
                w = 742
                h = 720
                top = (w-h)//2
                bottom = top+h
                image = cv2.resize(image, (w,h), interpolation=cv2.INTER_AREA)
                depth = cv2.resize(depth, (w,h), interpolation=cv2.INTER_NEAREST)
                da_depth = cv2.resize(da_depth, (w,h), interpolation=cv2.INTER_NEAREST)
                img_gray = cv2.resize(img_gray, (w,h), interpolation=cv2.INTER_AREA)
                

            voxel = eventcnt(event_stack, self.bins, h, w, pol=True)
            cnts = eventcnt(event_stack, 1, h, w, pol=False).squeeze()
            kernel = np.ones((11, 11), np.uint8)
            cnts_patch = convolve2d(cnts, kernel, mode='same', boundary='symm')
            # 计算图像的梯度（分别计算x和y方向）
            img_x = np.gradient(img_gray, axis=1)  # x方向的梯度
            img_y = np.gradient(img_gray, axis=0)  # y方向的梯度
            # 计算梯度幅值（合成梯度图）
            img_magnitude = np.sqrt(img_x**2 + img_y**2)
            # 归一化
            img_magnitude_norm = (img_magnitude-np.min(img_magnitude))/(np.max(img_magnitude)-np.min(img_magnitude))
            # candidate_points = np.argwhere(img_magnitude_norm >= 0.05)
            candidate = np.ones_like(img_magnitude_norm, dtype=np.bool_)
            candidate[img_magnitude_norm<0.1] = 0

            
            # print('cnts_patch',cnts_patch.max(),cnts_patch.min())
            candidate[cnts_patch<300]=0

            # valid_mask = (sparse_depth > 0)
            valid_mask = candidate
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            valid_mask =valid_mask.astype(np.uint8)*255
            valid_mask = cv2.dilate(valid_mask, kernel, iterations=1)
            valid_mask = valid_mask.astype(np.bool_)
        
        if self.norm:
            voxel = normalize_voxelgrid(voxel)
        
        
        h, w = depth.shape

        if self.mode == 'train':
            if self.train_crop:
                
                while True:
                    # print('depth',depth_path, depth.shape, depth.max(), depth.min())
                    if h > self.size[0]:
                        random_h = random.randint(0, h - self.size[0])
                    else:
                        random_h = 0
                        
                    if w > self.size[1]:
                        random_w = random.randint(0, w - self.size[1])
                    else:
                        random_w = 0
                
                    depth_tmp = self.crop(depth, random_h, random_w, self.size[0], self.size[1])
                    # print('depth_tmp',random_h,random_w,depth_tmp.shape,depth_tmp.max(),depth_tmp.min())
                    if depth_tmp.max() != depth_tmp.min():
                        break
                    
                image = self.crop(image, random_h, random_w, self.size[0], self.size[1])
                depth = self.crop(depth, random_h, random_w, self.size[0], self.size[1])
                da_depth = self.crop(da_depth, random_h, random_w, self.size[0], self.size[1])
                voxel = self.crop(voxel, random_h, random_w,self.size[0], self.size[1])
                valid_mask = self.crop(valid_mask,  random_h,random_w, self.size[0], self.size[1])

            else:
                image = cv2.resize(image, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
                depth = cv2.resize(depth, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)
                da_depth = cv2.resize(da_depth, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)
                voxel = cv2.resize(voxel, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)
                valid_mask = cv2.resize(valid_mask.astype(np.float32), (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)

            if random.random() < 0.5:
                voxel = np.flip(voxel, axis=1)
                depth = np.flip(depth, axis=1)
                valid_mask = np.flip(valid_mask, axis=1)
                image = np.flip(image, axis=1)
                da_depth = np.flip(da_depth, axis=1)
            if random.random() < 0.5:
                voxel = np.flip(voxel, axis=0)
                depth = np.flip(depth, axis=0)
                valid_mask = np.flip(valid_mask, axis=0)
                image = np.flip(image, axis=0)
                da_depth = np.flip(da_depth, axis=0)
        
        if self.mode == 'val':
            if h > self.size[0] or w > self.size[1]:
                image = cv2.resize(image, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)

                da_depth = cv2.resize(da_depth, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)
                voxel = cv2.resize(voxel, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)
                valid_mask = cv2.resize(valid_mask.astype(np.float32), (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)
            

        transform_dict = {'image': image}
        transform_dict['valid_mask'] = valid_mask
        transform_dict['depth'] = depth
        transform_dict['da_depth'] = da_depth
        sample = transform_dict
        
        
        sample['image'] = torch.from_numpy(sample['image'].copy()).permute(2,0,1)
        sample['depth'] = torch.from_numpy(sample['depth'].copy())
        sample['event_stack'] = torch.from_numpy(voxel.copy()).permute(2,0,1)
        sample['valid_mask'] = torch.from_numpy(sample['valid_mask'].copy())
        sample['da_depth'] = torch.from_numpy(sample['da_depth'].copy()).unsqueeze(0)
        sample['min_depth'] = min_depth
        sample['max_depth'] = max_depth

        
        return sample

    def __len__(self):
        return len(self.img_paths)