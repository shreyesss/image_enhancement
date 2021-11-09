import torch
import skimage.io as io
from skimage.transform import resize
from utils import get_model
import numpy as np
import torch.nn.functional as F


def post_process_disparity(disp):
    (_, h, w) = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def warp(img_left,img_right,model):
    left = torch.Tensor(img_left).permute(2,0,1).unsqueeze(0)
    right = torch.Tensor(img_right).permute(2,0,1).unsqueeze(0)
    inp = torch.cat([left,right],dim=1)
    out = model(inp)
    disp = out[0]
    left_disp = disp[0][0]
    right_disp = disp[0][1]
    # io.imsave("left_disp.png",left_disp.detach().numpy()*512)
    # io.imsave("right_disp.png",right_disp.detach().numpy()*512)
    warped_left = apply_disparity(right,-left_disp)
    warped_left = warped_left[0].detach().numpy().transpose(1,2,0)*255
    warped_right = apply_disparity(left,right_disp)
    warped_right = warped_right[0].detach().numpy().transpose(1,2,0)*255
    return warped_left,warped_right

def warp_post_proccess(img_left,img_right,model):
    left = torch.Tensor(img_left).permute(2,0,1).unsqueeze(0)
    right = torch.Tensor(img_right).permute(2,0,1).unsqueeze(0)
    inp = torch.cat([left,right],dim=1)
    out = model(inp)
    disp = post_process_disparity(out[0][0].detach().numpy())
    warped_left = apply_disparity(right,-torch.Tensor(disp))
    warped_left = warped_left[0].detach().numpy().transpose(1,2,0)*255
    warped_right = apply_disparity(left,torch.Tensor(disp))
    warped_right = warped_right[0].detach().numpy().transpose(1,2,0)*255

    return warped_left,warped_right
    


def p(img):
    print(img.min())
    print(img.max())
    print(img.mean())


def apply_disparity(img, disp):
        batch_size, _, height, width = img.shape

        # Original coordinates of pixels
        x_base = torch.linspace(0, 1, width).repeat(batch_size,
                    height, 1).type_as(img)
        y_base = torch.linspace(0, 1, height).repeat(batch_size,
                    width, 1).transpose(1, 2).type_as(img)

        # Apply shift in X direction
        x_shifts = disp#[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        # In grid_sample coordinates are assumed to be between -1 and 1
        output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',
                               padding_mode='zeros')

        return output

if __name__ == "__main__" :
    img_name = "00064"

    model = get_model(model = "resnet50",input_channels=6, pretrained=False)
    model_light = get_model(model = "resnet50",input_channels=6, pretrained=False)
    model.load_state_dict(torch.load("./checkpoints/low_stereo/stereo.pth"))
    model_light.load_state_dict(torch.load("/home/shreyas/MonoDepth-PyTorch/checkpoints/stereo/stereo.pth"))

    img_left = resize(io.imread(f"/media/shreyas/low_light_kitti2015/left_low/{img_name}.png"),(256,512))
    img_right = resize(io.imread(f"/media/shreyas/low_light_kitti2015/right_low/{img_name}.png"),(256,512))
    img_left_high = resize(io.imread(f"/media/shreyas/low_light_kitti2015/left_high/{img_name}.png"),(256,512))
    img_right_high = resize(io.imread(f"/media/shreyas/low_light_kitti2015/right_high/{img_name}.png"),(256,512))

   

    # warped_left,warped_right = warp(img_left,img_right,model)
    # warped_left_low,warped_right_low = warp(img_left_high,img_right_high,model)
    # warped_left_light,warped_right_light = warp(img_left_high,img_right_high,model_light)
    # img_left = img_left * 255
    # img_left_high = img_left_high * 255
    # img_right_high = img_right_high * 255
    warped_left,warped_right = warp_post_proccess(img_left,img_right,model)
    warped_left_low,warped_right_low = warp_post_proccess(img_left_high,img_right_high,model)
    warped_left_light,warped_right_light = warp_post_proccess(img_left_high,img_right_high,model_light)
    img_left = img_left * 255
    img_left_high = img_left_high * 255
    img_right_high = img_right_high * 255

    # p(img_left)
    # p(warped_left_low)
    # p(warped_left_light)



    io.imsave(f"r2l_warp_post_{img_name}.png",np.concatenate([warped_left,warped_left_low,warped_left_light,img_left_high],axis=0))
    io.imsave(f"l2r_warp_post_{img_name}.png",np.concatenate([warped_right,warped_right_low,warped_right_light,img_right_high],axis=0))
# io.imsave("disp2.png",disp*512)



