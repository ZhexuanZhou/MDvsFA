from models.discriminator import Discriminator
from models.generator1_can8 import Generator1_CAN8
from models.generator2_ucan64 import Generator2_UCAN64
from tools.dataloader import MyDataset
from tools.log import initialize_logger
from tools.fmeasure import calculateF1Measure
# from loss_funcions.loss import *

import cv2
import logging
import numpy as np 
from tqdm import tqdm 
import datetime

import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

import glob

composed = transforms.Compose(
    [
        transforms.Grayscale(1),
        transforms.ToTensor()
    ]
)
test_dataset = MyDataset('./data/test_org/*', './data/test_gt/*', transform=composed)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=8)

disc_net = nn.DataParallel(Discriminator()).cuda()
gen1_net = nn.DataParallel(Generator1_CAN8()).cuda()
gen2_net = nn.DataParallel(Generator2_UCAN64()).cuda()


disc_net_weight = './saved_models/discriminator_epoch_30.pth'
disc_net.load_state_dict(torch.load(disc_net_weight))
gen1_net_weight = './saved_models/generator1_epoch_30.pth'
gen1_net.load_state_dict(torch.load(gen1_net_weight))
gen2_net_weight = './saved_models/generator2_epoch_30.pth'
gen2_net.load_state_dict(torch.load(gen2_net_weight))

filenames = sorted(glob.glob('./data/test_org/*.png'))
filenames = [x.split('/')[-1] for x in filenames]

disc_net.eval()
gen1_net.eval()
gen2_net.eval()

with torch.no_grad():
    for ix, (x, y) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        img = x.to(device='cuda: 0', dtype=torch.float)
        mask = y.to(device='cuda: 0', dtype=torch.float)
        # G1
        g1_result = gen1_net(img)
        # output_image = torch.min(torch.max(result,minVar),maxVar)
        g1_output_image = g1_result.squeeze(0).cpu().numpy() * 255.0
        g1_output_image = np.rollaxis(g1_output_image, axis=0, start=3)
        cv2.imwrite('./test_results/{}_G1.png'.format(str(filenames[ix].replace('.png', ''))), \
            g1_output_image)

        # G2
        g2_result = gen2_net(img)
        # output_image = torch.min(torch.max(result,minVar),maxVar)
        g2_output_image = g2_result.squeeze(0).cpu().numpy() * 255.0
        g2_output_image = np.rollaxis(g2_output_image, axis=0, start=3)
        cv2.imwrite('./test_results/{}_G2.png'.format(str(filenames[ix].replace('.png', ''))), \
            g2_output_image)

        fusion = (g1_result+g2_result)/2
        cv2.imwrite('./test_results/{}_Res.png'.format(str(filenames[ix].replace('.png', ''))), \
            (g1_output_image+g2_output_image)/2)

