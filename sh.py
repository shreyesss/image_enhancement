import os
from PIL import Image
import torch
from torch.utils.data import Dataset  ,DataLoader
from transforms import image_transforms
import torch.nn as nn

class KittiLoader(Dataset):
    def __init__(self, root_dir, mode="train", transform=None):
        if mode == "train":
            self.left_paths_low = []
            self.right_paths_low = []
            self.left_paths_high = []
            self.right_paths_high = []
            
            left_dir_low = os.path.join(root_dir,"left_low")
            self.left_paths_low.extend([os.path.join(left_dir_low, fname) for fname in os.listdir(left_dir_low)])
                    
            right_dir_low = os.path.join(root_dir,"right_low")
            self.right_paths_low.extend([os.path.join(right_dir_low, fname) for fname in os.listdir(right_dir_low)])
            
            left_dir_high = os.path.join(root_dir,"left_high")
            self.left_paths_high.extend([os.path.join(left_dir_high, fname) for fname in os.listdir(left_dir_high)])
                    
            right_dir_high = os.path.join(root_dir,"right_high")
            self.right_paths_high.extend([os.path.join(right_dir_high, fname) for fname in os.listdir(right_dir_high)])
            
            self.left_paths_low = sorted(self.left_paths_low)#[:64]
            self.right_paths_low = sorted(self.right_paths_low)#[:64]
            self.left_paths_high = sorted(self.left_paths_high)#[:64]
            self.right_paths_high = sorted(self.right_paths_high)#[:64]

        if mode == "val" or mode == "test:":

            self.left_paths = []
            self.right_paths = []
            self.gt_paths = []

            
            left_dir_low = os.path.join(root_dir,"left_low")
            self.left_paths_low.extend([os.path.join(left_dir_low, fname) for fname in os.listdir(left_dir_low)])
                    
            right_dir_low = os.path.join(root_dir,"right_low")
            self.right_paths_low.extend([os.path.join(right_dir_low, fname) for fname in os.listdir(right_dir_low)])
            
            left_dir_high = os.path.join(root_dir,"left_high")
            self.left_paths_high.extend([os.path.join(left_dir_high, fname) for fname in os.listdir(left_dir_high)])
                    
            right_dir_high = os.path.join(root_dir,"right_high")
            self.right_paths_high.extend([os.path.join(right_dir_high, fname) for fname in os.listdir(right_dir_high)])

            self.left_paths_low = sorted(self.left_paths_low)#[:64]
            self.right_paths_low = sorted(self.right_paths_low)#[:64]
            self.left_paths_high = sorted(self.left_paths_high)#[:64]
            self.right_paths_high = sorted(self.right_paths_high)#[:64]
            assert len(self.right_paths) == len(self.left_paths)

        self.transform = transform
        self.mode = mode


    def __len__(self):
        return len(self.left_paths_low)

    def __getitem__(self, idx):
        
        if self.mode == 'train':
            left_image_low = Image.open(self.left_paths_low[idx])
            right_image_low = Image.open(self.right_paths_low[idx])
            left_image_high = Image.open(self.left_paths_high[idx])
            right_image_high = Image.open(self.right_paths_high[idx])
            #print(right_image)
            sample = {'left_image_low': left_image_low, 'right_image_low': right_image_low ,
                      'left_image_high' : left_image_high , 'right_image_high' : right_image_high}

            if self.transform:
                sample = self.transform(sample)
                return sample
            else:
                return sample

        if self.mode == 'val' or self.mode == 'test': 
            left_image_low = Image.open(self.left_paths_low[idx])
            right_image_low = Image.open(self.right_paths_low[idx])
            left_image_high = Image.open(self.left_paths_high[idx])
            right_image_high = Image.open(self.right_paths_high[idx])
            #print(right_image)
            sample = {'left_image_low': left_image_low, 'right_image_low': right_image_low ,
                      'left_image_high' : left_image_high , 'right_image_high' : right_image_high}


            if self.transform:
                sample = self.transform(sample)
                return sample
            else:
                return sample

        else:
            if self.transform:
                left_image = self.transform(left_image)
            return left_image


def SSIM(x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        return torch.mean(torch.clamp((1 - SSIM) / 2, 0, 1))


if __name__ == "__main__":
    import sys
    train_dir  = "/media/shreyas/low_light"
    #train_dir  = "/media/shreyas/low_light_kitti2015"
    data_transform = image_transforms(
        mode="train",
        augment_parameters=[
        0.8,
        1.2,
        0.5,
        2.0,
        0.8,
        1.2,
        ],
        do_augmentation=True,
        size = (256,512))
    ds = KittiLoader(train_dir,transform = data_transform )
    loader = DataLoader(ds, batch_size= 4,
                            shuffle=True, num_workers= 2,
                            pin_memory=True)
    for data in loader:
       
        #print(data["ground_truth"].size)
        inp = data["left_image_low"]
        out = data["right_image_high"]
        #print(inp[:,0,:,:].shape,out[:,0,:,:].shape)
        print(SSIM(inp[:,0,:,:],out[:,0,:,:]))
        sys.exit()
    #print(loader.right_paths[:5])
    #print(loader.gt_paths[:5])
    
    
