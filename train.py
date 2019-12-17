from models import Discriminator
from models import Generator1_CAN8
from models import Generator2_UCAN64
from tools import MyDataset
from tools import initialize_logger
from tools import calculateF1Measure
from tools import init_linear_weights, init_conv2_weights
from tools import para_parser
from loss_funcions import *

import cv2
import logging
import numpy as np 
from tqdm import tqdm 
import time
import datetime
import sys 
import glob
import os
import math

import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

initialize_logger('./logs')

args = para_parser()

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# Models
disc_net = Discriminator()
gen1_net = Generator1_CAN8(is_anm=args.is_anm)
gen2_net = Generator2_UCAN64(is_anm=args.is_anm)

if cuda:
    disc_net.to(device='cuda: 0')
    gen1_net.to(device='cuda: 0')
    gen2_net.to(device='cuda: 0')

if args.parallel and cuda:
    disc_net = torch.nn.DataParallel(Discriminator())
    gen1_net = torch.nn.DataParallel(Generator1_CAN8(is_anm=args.is_anm))
    gen2_net = torch.nn.DataParallel(Generator2_UCAN64(is_anm=args.is_anm))

    disc_net.to(device='cuda: 0')
    gen1_net.to(device='cuda: 0')
    gen2_net.to(device='cuda: 0')


# weight init
disc_net.apply(init_linear_weights)
gen1_net.apply(init_conv2_weights)
gen2_net.apply(init_conv2_weights)

# optimizers
disc_optimizer = torch.optim.Adam(disc_net.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
gen1_optimizer = torch.optim.Adam(gen1_net.parameters(), lr=args.g1_lr, betas=(0.9, 0.999))
gen2_optimizer = torch.optim.Adam(gen2_net.parameters(), lr=args.g2_lr, betas=(0.9, 0.999))

# dataset
logging.info("Preparing dataset...")

composed = transforms.Compose(
    [
        transforms.Grayscale(1),
        transforms.ToTensor(),
    ])

train_dataset = MyDataset(args.training_imgs, args.training_masks, transform=composed)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

test_dataset = MyDataset(args.evl_imgs, args.evl_masks, transform=composed)
test_dataloader = DataLoader(test_dataset,batch_size=1, shuffle=False)

train_dataloader_len = len(train_dataloader)

# training phases
logging.info("Start training...")

for epoch in range(1, args.epochs+1):
    disc_net.train()
    gen1_net.train()
    gen2_net.train()
    for ix, (xs, ys) in \
        tqdm(enumerate(train_dataloader), total=train_dataloader_len):
       
        N = xs.size(0)
        ones = torch.FloatTensor(N, 1).fill_(1.0).type(FloatTensor)
        zeros = torch.FloatTensor(N, 1).fill_(0.0).type(FloatTensor)
        gen_gt1 = torch.cat([ones,zeros,zeros],1)
        gen_gt2 = torch.cat([zeros,ones,zeros],1)
        gen_gt3 = torch.cat([zeros,zeros,ones],1)

        imgs = xs.type(FloatTensor)
        masks = ys.type(FloatTensor)

        ## train discriminator
        disc_optimizer.zero_grad()

        g1_output = gen1_net(imgs)    
        g2_output = gen2_net(imgs)

        d_input1 = torch.cat([imgs, masks*2-1], dim=1)
        d_input2 = torch.cat([imgs, g1_output.detach()*2-1], dim=1)
        d_input3 = torch.cat([imgs, g2_output.detach()*2-1], dim=1)

        d_result1, _ = disc_net(d_input1)

        d_loss1 = disc_criterion(d_result1, gen_gt1)
        
        d_result2, feat_map2 = disc_net(d_input2)
        d_loss2 = disc_criterion(d_result2, gen_gt2)

        d_result3, feat_map3 = disc_net(d_input3)
        d_loss3 = disc_criterion(d_result3, gen_gt3)
        
        d_loss = d_loss1 + d_loss2 + d_loss3

        d_loss.backward()
        disc_optimizer.step()

        # trainig G1 and G2 separately
        gen1_optimizer.zero_grad()
        gen2_optimizer.zero_grad()

        g1_output = gen1_net(imgs)
        g2_output = gen2_net(imgs)

        d_input2 = torch.cat([imgs, g1_output*2-1], dim=1)
        d_input3 = torch.cat([imgs, g2_output*2-1], dim=1)

        d_result2, feat_map2 = disc_net(d_input2)
        d_result3, feat_map3 = disc_net(d_input3)

        MD1, FA1, MF_loss1 = MF1_criterion(g1_output, masks)
        gen_adv_loss1 = adv_loss(d_result2, gen_gt1)

        gc2_loss = gc_criterion(feat_map2, feat_map3.detach())      
        gen1_loss = 100*MF_loss1 + 10*gen_adv_loss1 + gc2_loss
        
        MD2, FA2, MF_loss2 = MF2_criterion(g2_output, masks)
        gen_adv_loss2 = adv_loss(d_result3, gen_gt1)

        gc3_loss = gc_criterion(feat_map3, feat_map2.detach())
        gen2_loss = 100*MF_loss2 + 10*gen_adv_loss2  + gc3_loss
        
        gen1_loss.backward(retain_graph=True)
        gen2_loss.backward()
        gen1_optimizer.step()
        gen2_optimizer.step()

        if ix % 100 == 0 or ix == train_dataloader_len:
            logging.info('===============================================')
            logging.info('Epoch: {}/{}, Batch: {}/{}'.format(epoch, args.epochs, ix + 1, train_dataloader_len))
            logging.info('===Discriminator===============================')
            logging.info('d_loss: {}'.format(d_loss.item()))
            logging.info('===Generator1==================================')
            logging.info('MD1: {}'.format(MD1.item()))
            logging.info('FA1: {}'.format(FA1.item()))
            logging.info('MF_loss1: {}'.format(MF_loss1.item()))
            logging.info('gc_loss1: {}'.format(gc2_loss.item()))
            logging.info('gen_adv_loss1: {}'.format(gen_adv_loss1.item()))
            logging.info('gen1_loss: {}'.format(gen1_loss.item()))
            logging.info('===Generator2=================================')
            logging.info('MD2: {}'.format(MD2.item()))
            logging.info('FA2: {}'.format(FA2.item()))
            logging.info('MF_loss2: {}'.format(MF_loss2.item()))
            logging.info('gc_loss2: {}'.format(gc3_loss.item()))
            logging.info('gen_adv_loss2: {}'.format(gen_adv_loss2.item()))
            logging.info('gen2_loss: {}'.format(gen2_loss.item()))

    if epoch % 1 == 0: 
        torch.save(gen1_net.state_dict(), "saved_models/generator1_epoch_%d.pth" % (epoch))
        torch.save(gen2_net.state_dict(), "saved_models/generator2_epoch_%d.pth" % (epoch))
        torch.save(disc_net.state_dict(), "saved_models/discriminator_epoch_%d.pth" % (epoch))
    
        filenames = sorted(glob.glob('./data/test_org/*.png'))
        filenames = [x.split('/')[-1] for x in filenames]
       
        disc_net.eval()
        gen1_net.eval()
        gen2_net.eval()
        with torch.no_grad():
            sum_val_loss_g1 = 0
            sum_val_false_ratio_g1 = 0 
            sum_val_detect_ratio_g1 = 0

            sum_val_F1_g1 = 0
            g1_time = 0

            sum_val_loss_g2 = 0
            sum_val_false_ratio_g2 = 0 
            sum_val_detect_ratio_g2 = 0

            sum_val_F1_g2 = 0
            g2_time = 0

            sum_val_loss_g3 = 0
            sum_val_false_ratio_g3 = 0 
            sum_val_detect_ratio_g3 = 0

            sum_val_F1_g3 = 0
            for ix, (x, y) in tqdm(enumerate(test_dataloader), 
                total=len(test_dataloader)):

                img = x.to(device='cuda: 0', dtype=torch.float)
                mask = y.to(device='cuda: 0', dtype=torch.float)
                # G1
                g1_result = gen1_net(img)
                g1_output_image = g1_result.squeeze(0).cpu().numpy() * 255.0
                g1_output_image = np.rollaxis(g1_output_image, axis=0, start=3)
                cv2.imwrite('./training_results/{}_G1.png'.format(\
                    filenames[ix]
                        ), g1_output_image)

                # G2
                g2_result = gen2_net(img)
                g2_output_image = g2_result.squeeze(0).cpu().numpy() * 255.0
                g2_output_image = np.rollaxis(g2_output_image, axis=0, start=3)
                cv2.imwrite('./training_results/{}_G2.png'.format(\
                    filenames[ix]
                        ), g2_output_image)

                fusion = (g1_result+g2_result)/2
                cv2.imwrite('./training_results/{}_Res.png'.format(\
                    filenames[ix]
                        ), (g1_output_image+g2_output_image)/2)

                minVar = torch.zeros_like(mask)
                maxVar = torch.ones_like(mask)
                # calculate G1 Fmeasure score
                result1 = gen1_net(img)
                train_loss = torch.mean((result1-mask)**2)
                sum_val_loss_g1 += train_loss.item()
                train_false_ratio = torch.mean(torch.max(minVar,result1 - mask))
                sum_val_false_ratio_g1 += train_false_ratio.item()

                train_detect_ratio = torch.sum(result1*mask)/torch.max(mask.sum(),maxVar)
                sum_val_detect_ratio_g1 += torch.mean(train_detect_ratio)
                train_F1 = calculateF1Measure(result1.cpu().numpy(),mask.cpu().numpy(),0.5)
                sum_val_F1_g1 += train_F1

                # calculate G2 Fmeasure score
                result2 = gen2_net(img)
                train_loss = torch.mean((result2-mask)**2)
                sum_val_loss_g2 += train_loss.item()
                train_false_ratio = torch.mean(torch.max(minVar,result2 - mask))
                sum_val_false_ratio_g2 += train_false_ratio.item()

                train_detect_ratio = torch.sum(result2*mask)/torch.max(mask.sum(),maxVar)
                sum_val_detect_ratio_g2 += torch.mean(train_detect_ratio)
                train_F2 = calculateF1Measure(result2.cpu().numpy(),mask.cpu().numpy(),0.5)
                sum_val_F1_g2 += train_F2

                # calculate Fusion result Fmeasure score
                result = (result1 + result2)/2
                train_loss = torch.mean((result-mask)**2)
                sum_val_loss_g3 += train_loss.item()
                train_false_ratio = torch.mean(torch.max(minVar,result - mask))
                sum_val_false_ratio_g3 += train_false_ratio.item()

                train_detect_ratio = torch.sum(result*mask)/torch.max(mask.sum(),maxVar)
                sum_val_detect_ratio_g3 += torch.mean(train_detect_ratio)
                train_F3 = calculateF1Measure(result.cpu().numpy(),mask.cpu().numpy(),0.5)
                sum_val_F1_g3 += train_F3
            logging.info("Epoch {} evaluation:".format(epoch))
            logging.info("==========g1 results ============================")
            avg_val_loss_g1 = sum_val_loss_g1/100
            avg_val_false_ratio_g1  = sum_val_false_ratio_g1/100
            avg_val_detect_ratio_g1 = sum_val_detect_ratio_g1/100
            avg_val_F1_g1 = sum_val_F1_g1/100

            logging.info("==========val_L2_loss is %f"% (avg_val_loss_g1))
            logging.info("==========falseAlarm_rate is %f"% (avg_val_false_ratio_g1))
            logging.info("==========detection_rate is %f"% (avg_val_detect_ratio_g1))
            logging.info("==========F1 measure is %f"% (avg_val_F1_g1))

            logging.info("========== g2 results ============================")
            avg_val_loss_g2 = sum_val_loss_g2/100
            avg_val_false_ratio_g2  = sum_val_false_ratio_g2/100
            avg_val_detect_ratio_g2 = sum_val_detect_ratio_g2/100
            avg_val_F1_g2 = sum_val_F1_g2/100

            logging.info("==========val_L2_loss is %f"% (avg_val_loss_g2))
            logging.info("==========falseAlarm_rate is %f"% (avg_val_false_ratio_g2))
            logging.info("==========detection_rate is %f"% (avg_val_detect_ratio_g2))
            logging.info("==========F1 measure is %f"% (avg_val_F1_g2))

            logging.info("========== g3 results ============================")
            avg_val_loss_g3 = sum_val_loss_g3/100
            avg_val_false_ratio_g3  = sum_val_false_ratio_g3/100
            avg_val_detect_ratio_g3 = sum_val_detect_ratio_g3/100
            avg_val_F1_g3 = sum_val_F1_g3/100

            logging.info("==========val_L2_loss is %f"% (avg_val_loss_g3))
            logging.info("==========falseAlarm_rate is %f"% (avg_val_false_ratio_g3))
            logging.info("==========detection_rate is %f"% (avg_val_detect_ratio_g3))
            logging.info("==========F1 measure is %f"% (avg_val_F1_g3))
