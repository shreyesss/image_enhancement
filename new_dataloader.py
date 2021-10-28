import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from transforms import image_transforms

class KittiLoader(Dataset):
    def __init__(self, root_dir, mode="train", transform=None):
        if mode == "train":
            self.left_paths = []
            self.right_paths = []
            
            left_dir = os.path.join(root_dir,"left_low")
            self.left_paths.extend([os.path.join(left_dir, fname) for fname in os.listdir(left_dir)])
                    
            right_dir = os.path.join(root_dir,"right_low")
            self.right_paths.extend([os.path.join(right_dir, fname) for fname in os.listdir(right_dir)])
            self.left_paths = sorted(self.left_paths)#[:64]
            self.right_paths = sorted(self.right_paths)#[:64]

        if mode == "val" or mode == "test:":

            self.left_paths = []
            self.right_paths = []
            self.gt_paths = []

            
            left_dir = os.path.join(root_dir,"left_low")
            self.left_paths.extend([os.path.join(left_dir, fname) for fname in os.listdir(left_dir)])
                    
            right_dir = os.path.join(root_dir,"right_low")
            self.right_paths.extend([os.path.join(right_dir, fname) for fname in os.listdir(right_dir)])
            
            left_gt_dir = os.path.join(root_dir,"left_gt")
            self.gt_paths.extend([os.path.join(left_gt_dir, fname) for fname in os.listdir(left_gt_dir)])

            self.left_paths = sorted(self.left_paths) [:100]
            self.right_paths = sorted(self.right_paths)[:100]
            self.gt_paths = sorted(self.gt_paths)[:100]
            assert len(self.right_paths) == len(self.left_paths)

        self.transform = transform
        self.mode = mode


    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        
        if self.mode == 'train':
            left_image = Image.open(self.left_paths[idx])
            right_image = Image.open(self.right_paths[idx])
            #print(right_image)
            sample = {'left_image': left_image, 'right_image': right_image}

            if self.transform:
                sample = self.transform(sample)
                return sample
            else:
                return sample

        if self.mode == 'val': 
            left_image = Image.open(self.left_paths[idx])
            right_image = Image.open(self.right_paths[idx])
            #print(right_image)
            gt = Image.open(self.gt_paths[idx])
            sample = {'left_image': left_image, 'right_image': right_image, 'ground_truth':gt}

            if self.transform:
                sample = self.transform(sample)
                return sample
            else:
                return sample

        else:
            if self.transform:
                left_image = self.transform(left_image)
            return left_image


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
    loader = KittiLoader(train_dir, mode="train",transform=data_transform)
    for data in loader:
        
        #print(data["ground_truth"].size)
        print(data["left_image"].shape)
        print(data["right_image"].shape)
        sys.exit()
    #print(loader.right_paths[:5])
    #print(loader.gt_paths[:5])
    
    