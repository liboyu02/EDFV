import random

import numpy
import torch
import cv2
import numpy as np
import traceback

def fix_random_seed(seed: int):
    

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def nan_hook(self, inp, output):
    if isinstance(output, dict):
        outputs = output.values()
        # keys = output.keys()
    if not isinstance(output, tuple):
        outputs = [output]
    else:
        outputs = output
    # print(outputs)
    
    for i, out in enumerate(outputs):
        if torch.is_tensor(out):
            nan_mask = torch.isnan(out)
            if nan_mask.any():
                print("In", self.__class__.__name__)
                traceback.print_stack()
                raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])
            
def register_nan_detection_hook(model):
    
    for name, param in model.named_parameters():
        def detect_nan(grad, name=name):
            if torch.isnan(grad).any():
                traceback.print_stack()
                raise RuntimeError(f"NaN detected in {name}! Stopping the program.")
                # sys.exit(1)
        param.register_hook(detect_nan)

def colorize_cv2(depth):  
       
       depth = (depth - depth.min()) / (depth.max() - depth.min())
    #    print('depth',depth.shape)
       if isinstance(depth, torch.Tensor):
          depth = depth.cpu().numpy()
       vis_depth = cv2.applyColorMap((depth*255).astype(np.uint8), cv2.COLORMAP_JET)
    #    print('vis_depth',vis_depth.shape)
       vis_depth = vis_depth.astype(np.float64)/255.0
       return vis_depth
    #    return vis_depth

def colorize_gt_cv2(depth, gt_depth):
        depth = (depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min())
        depth = np.clip(depth, 0, 1)
        vis_depth = cv2.applyColorMap((depth*255).astype(np.uint8), cv2.COLORMAP_JET)
    #    print('vis_depth',vis_depth.shape)
        vis_depth = vis_depth.astype(np.float64)/255.0
        return vis_depth

def colorize_cv2_valid(depth):  
       
       depth = (depth - depth[depth>0].min()) / (depth[depth>0].max() - depth[depth>0].min())
       depth = np.clip(depth, 0, 1)
    #    print('depth',depth.shape)
       if isinstance(depth, torch.Tensor):
          depth = depth.cpu().numpy()
       vis_depth = cv2.applyColorMap((depth*255).astype(np.uint8), cv2.COLORMAP_JET)
    #    print('vis_depth',vis_depth.shape)
       vis_depth = vis_depth.astype(np.float64)/255.0
       return vis_depth