import argparse
import time
import torch
import numpy as np
import torch.optim as optim
import os
import gc
# custom modules

from loss import MonodepthLoss
from utils import get_model, to_device, prepare_dataloader
from metrics import compute_metrics
# plot params

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (15, 10)


from tensorboardX import SummaryWriter


def return_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Monodepth')

    parser.add_argument('--data_dir',
                        help='path to the dataset folder. \
                        It should contain subfolders with following structure:\
                        "image_02/data" for left images and \
                        "image_03/data" for right images', default = "/media/shreyas/low_light/"
                        )
    parser.add_argument('--val_data_dir',
                        help='path to the validation dataset folder. \
                            It should contain subfolders with following structure:\
                            "image_02/data" for left images and \
                            "image_03/data" for right images',default = "/media/shreyas/low_light_kitti2015/"
                        )
    parser.add_argument('--model_path', help='path to the trained model',default="./checkpoints/low_stereo/stereo_cpt.pth")
    parser.add_argument('--output_directory',
                        help='where save dispairities\
                        for tested images'
                        ,default=None)
    parser.add_argument('--input_height', type=int, help='input height',
                        default=256)
    parser.add_argument('--input_width', type=int, help='input width',
                        default=512)
    parser.add_argument('--model', default='resnet50',
                        help='encoder architecture: ' +
                        'resnet18_md or resnet50_md ' + '(default: resnet18)'
                        + 'or torchvision version of any resnet model'
                        )
    parser.add_argument('--pretrained', default=True,
                        help='Use weights of pretrained model'
                        )
    parser.add_argument('--mode', default='test',
                        help='mode: train or test (default: train)')
    parser.add_argument('--epochs', default=50,
                        help='number of total epochs to run')
    parser.add_argument('--learning_rate', default=1e-4,
                        help='initial learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', default=8,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--adjust_lr', default=True,
                        help='apply learning rate decay or not\
                        (default: True)'
                        )
    parser.add_argument('--device',
                        default='cuda:0',
                        help='choose cpu or cuda:0 device"'
                        )
    parser.add_argument('--do_augmentation', default=True,
                        help='do augmentation of images or not')
    parser.add_argument('--augment_parameters', default=[
        0.8,
        1.2,
        0.5,
        2.0,
        0.8,
        1.2,
        ],
            help='lowest and highest values for gamma,\
                        brightness and color respectively'
            )
    parser.add_argument('--print_images', default=False,
                        help='print disparity and image\
                        generated from disparity on every iteration'
                        )
    parser.add_argument('--print_weights', default=False,
                        help='print weights of every layer')
    parser.add_argument('--input_channels', default=6,
                        help='Number of channels in input tensor')
    parser.add_argument('--num_workers', default=4,
                        help='Number of workers in dataloader')
                        
    parser.add_argument('--logdir', default="./checkpoints/low_stereo",
                        help='tensorboard directory logs')
    parser.add_argument('--use_multiple_gpu', default=False)
    parser.add_argument('--do_stereo', default=True)
    args = parser.parse_args()
    return args


def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Sets the learning rate to the initial LR\
        decayed by 2 every 10 epochs after 30 epoches"""

    if epoch >= 30 and epoch < 40:
        lr = learning_rate / 2
    elif epoch >= 40:
        lr = learning_rate / 4
    else:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def post_process_disparity(disp):
    (_, h, w) = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


class Model:

    def __init__(self, args):
        self.args = args

        os.makedirs(args.logdir, exist_ok=True)
        # create summary logger
        print("creating new summary file")
        self.logger = SummaryWriter(args.logdir)
        # Set up model
        self.device = args.device
        self.model = get_model(args.model, input_channels=args.input_channels, pretrained=args.pretrained)
        self.model = self.model.to(self.device)
        if args.use_multiple_gpu:
            self.model = torch.nn.DataParallel(self.model)

        if args.mode == 'train':
            self.loss_function = MonodepthLoss(
                n=4,
                SSIM_w=0.85,
                disp_gradient_w=0.1, lr_w=1).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=args.learning_rate)
            self.val_n_img, self.val_loader = prepare_dataloader(args.val_data_dir, args.mode,
                                                                 args.augment_parameters,
                                                                 False, args.batch_size,
                                                                 (args.input_height, args.input_width),
                                                                 args.num_workers,True)
        else:
            self.model.load_state_dict(torch.load(args.model_path))
            args.augment_parameters = None
            args.do_augmentation = False
            args.batch_size = 1

        # Load data
        self.output_directory = args.output_directory
        self.input_height = args.input_height
        self.input_width = args.input_width

        self.n_img, self.loader = prepare_dataloader(args.data_dir, args.mode, args.augment_parameters,
                                                     args.do_augmentation, args.batch_size,
                                                     (args.input_height, args.input_width),
                                                     args.num_workers,False)


        if 'cuda' in self.device:
            torch.cuda.synchronize()


    def train(self):
        losses = []
        val_losses = []
        best_loss = float('Inf')
        best_val_loss = float('Inf')

        running_val_loss = 0.0


        for epoch in range(self.args.epochs):

            
            ##log val metrics
            if self.args.adjust_lr:
                adjust_learning_rate(self.optimizer, epoch,
                                     self.args.learning_rate)
            c_time = time.time()
            running_loss = 0.0
            self.model.train()
            for batch_idx, data in enumerate(self.loader):
                global_step = epoch * len(self.loader) + batch_idx
                # Load data
                data = to_device(data, self.device)
                left = data['left_image']
                right = data['right_image']
                print(left.shape)

                # One optimization iteration
                self.optimizer.zero_grad()

                if self.args.do_stereo == True:
                    input_ = torch.cat((left,right),dim=1)
                else:
                    input_ = left 
                disps = self.model(input_)
                loss = self.loss_function(disps, [left, right])
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                print("loss this batch:", loss.item())
                running_loss += loss.item()

                del disps,data
                # log train metrics during steps -- train_iter
                self.logger.add_scalar("train_batch_loss/train_iter", loss.item(), global_step)
            
            gc.collect()
            # Estimate loss per image
            # log train_epohs wise metrics


            running_val_loss = 0.0
            #self.model.eval()
            running_metrics = {}
            running_metrics["d1_all"] = 0
            running_metrics["abs_rel"] = 0
            running_metrics["sq_rel"] = 0
            running_metrics["rms"] = 0 
            running_metrics["log_rms"] = 0
            running_metrics["a1"] = 0
            running_metrics["a2"] = 0
            running_metrics["a3"] = 0
            running_metrics["N"] = 0
            for data in self.val_loader:
                data = to_device(data, self.device)
                left = data['left_image']
                right = data['right_image']
                gt = data["ground_truth"]
                if self.args.do_stereo == True:
                    input_ = torch.cat((left,right),dim=1)
                else:
                    input_ = left 
                disps = self.model(input_)
                loss = self.loss_function(disps, [left, right])
                metrics = compute_metrics(disps[0].detach().cpu().numpy(),gt.detach().cpu().numpy())
                val_losses.append(loss.item())
                running_val_loss += loss.item()
                for key,val in enumerate(metrics):
                    #print(key,val)
                    running_metrics[val] = running_metrics[val] + metrics[val]

            running_N = running_metrics["N"]
            for key,val in enumerate(running_metrics):
                running_metrics[val] = running_metrics[val]/running_N

            running_val_loss /= self.val_n_img / self.args.batch_size
            self.logger.add_scalar("Validation_metrics/val_loss", running_val_loss, epoch)

            for key,value in enumerate(running_metrics):
                self.logger.add_scalar("Validation_metrics/{}".format(value), running_metrics[value], epoch)

            running_loss /= self.n_img / self.args.batch_size
            self.logger.add_scalar("Avg_epoch_loss/train_epoch", running_loss, epoch)
            print (
                'Epoch:',
                epoch + 1,
                'train_loss:',
                running_loss,
                'time:',
                round(time.time() - c_time, 3),
                's',
                )
            
            self.save(self.args.model_path[:-4] + '_last.pth')
            if running_val_loss < best_val_loss:
                self.save(self.args.model_path[:-4] + '_cpt.pth')
                best_val_loss = running_val_loss
                print('Model_saved')

        print ('Finished Training. Best loss:', best_loss)
        self.save(self.args.model_path)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def test(self):
        self.model.eval()
        disparities = np.zeros((self.n_img,
                               self.input_height, self.input_width),
                               dtype=np.float32)
        #disparities_pp = np.zeros((self.n_img,
        #                          self.input_height, self.input_width),
        #                          dtype=np.float32)
        with torch.no_grad():
            for (i, data) in enumerate(self.loader):
                # Get the inputs
                data = to_device(data, self.device)
                
                left = data.squeeze()
                # Do a forward pass
                disps = self.model(left)
                disp = disps[0][:, 0, :, :].unsqueeze(1)
                disparities[i] = disp[0].squeeze().cpu().numpy()
                #disparities_pp[i] = \
                #    post_process_disparity(disps[0][:, 0, :, :]\
                #                           .cpu().numpy())

        #np.save(self.output_directory + '/disparities.npy', disparities)
        #np.save(self.output_directory + '/disparities_pp.npy',
                disparities_pp)
        print('Finished Testing')


def main(args):
    
    if args.mode == 'train':
        model = Model(args)
        model.train()
    elif args.mode == 'test':
        model_test = Model(args)
        model_test.test()


if __name__ == '__main__':
    args = return_arguments()
    main(args)

