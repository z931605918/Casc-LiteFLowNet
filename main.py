from __future__ import print_function
import cv2

cv2.setNumThreads(0)
import sys
import pdb
import argparse
import collections
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import re
#from utils.flowlib import flow_to_image
from utils import logger
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.dataloader import MyDataset
from utils.augmentations import Augmentation, Basetransform
torch.backends.cudnn.benchmark = True
from utils.multiscaleloss import MultiscaleLoss, realEPE, RMSE
from glob import glob
import cv2 as cv
from tqdm import tqdm




from visdom import Visdom

#python -m visdom.server
def find_NewFile(path):
    # 获取文件夹中的所有文�?
    lists = glob(os.path.join(path, '*.tar'))
    # 对获取的文件根据修改时间进行排序
    lists.sort(key=lambda x: os.path.getmtime(x))
    # 把目录和文件名合成一个路�?
    file_new = lists[-1]
    return file_new

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=256,
                    help='maxium disparity, out of range pixels will be masked out. Only affect the coarsest cost volume size')
parser.add_argument('--fac', type=float, default=1,
                    help='controls the shape of search grid. Only affect the coarsest cost volume size')
parser.add_argument('--logname', default='logname_en',
                    help='name of the log file')
parser.add_argument('--casc_logname', default='logname_casc_en_four_layer',
                    help='name of the log file')
parser.add_argument('--database', default='/',
                    help='path to the database')
parser.add_argument('--epochs', type=int, default=1500,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default=None,
                    help='path of the pre-trained model')
parser.add_argument('--savemodel', default='./',
                    help='path to save the model')
parser.add_argument('--or_resume', default='D:\desktop\Lite-Flownet-master\logname_old\\finetune_1272.tar',  #在此加载基础模型
                    help='whether to reset moving mean / other hyperparameters')
#parser.add_argument('--casc_resume',default=None)
parser.add_argument('--casc_resume',default='D:\desktop\TCN-flownet\程序\Casc-LiteFlownet\\logname_casc_four_layer\\finetune_57.tar')
#parser.add_argument('--casc_resume', default='D:\\desktop\\TCN-flownet\\程序\\Casc-LiteFlownet\\logname_casc_en_re\\finetune_363.tar', #在此加载串联模型
                   # help='whether to reset moving mean / other hyperparameters')
parser.add_argument('--stage', default='chairs',
                    help='one of {chairs, things, 2015train, 2015trainval, sinteltrain, sinteltrainval}')
parser.add_argument('--ngpus', type=int, default=8,
                    help='number of gpus to use.')
args = parser.parse_args()
#python -m visdom.server
baselr = 1e-5  #0.00001/2=0.000005=5e-6
batch_size = 1


torch.cuda.set_device(0)

dataset = MyDataset('D:\desktop\连续流场数据集\datasets',
                    transform=Augmentation(size=256, mean=(128)))

from lite_flownet import liteflownet #基础模型
print('%d batches per epoch' % (len(dataset) // batch_size))
from casc_flownet_en_3layer import casc_liteflownet as casc_en_3layer
from casc_flownet_en_4layer import casc_liteflownet as casc_en_4layer
model=liteflownet(args.or_resume)
#casc_model = casc_1(args.casc_resume)
casc_model=casc_en_4layer(args.casc_resume)
#python -m visdom.server
model.cuda()
casc_model.cuda()
#summary(model, input_size=(3, 3, 256, 256))
#optimizer = optim.Adam(filter(lambda p:p.requires_grad, casc_model.parameters()), lr=baselr, betas=(0.9, 0.999), amsgrad=False)
optimizer = optim.Adam(casc_model.parameters(), lr=baselr, betas=(0.9, 0.999), amsgrad=False)
#optimizer_or = optim.Adam(model.parameters(), lr=baselr, betas=(0.9, 0.999), amsgrad=False)
# optimizer = optim.SGD(model.parameters(), lr=baselr, momentum=0.9,  weight_decay=5e-4)

criterion = MultiscaleLoss()
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True,min_lr=1e-8)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, eta_min=5e-8)

#TestImgLoader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=2,
                       #                     drop_last=True, pin_memory=True)




def train(imgL, imgR, flowl0):
    casc_model.train()
    model.eval()
    Im1=imgL[0].cuda()
    Im2=imgR[0].cuda()
    flowl0[0]=flowl0[0].cuda()
    outs=[model((Im1,Im2))]
    num=1
    total_loss = 0
    optimizer.zero_grad()
    rmse_1=RMSE(outs[0].detach(),flowl0[0].detach())
    for I1,I2,flo in zip(imgL[1:],imgR[1:],flowl0[1:]):
        I1, I2, flo = I1.cuda(), I2.cuda() ,flo.cuda()
        last_out=outs[-1]
        output=casc_model((I1,I2),last_out)
        total_loss+=criterion.forward(output,flo)
        outs.append(output[0])
        num+=1
    mean_loss=total_loss/6
    mean_loss.backward()
    optimizer.step()
    vis = {}
    mean_RMSE=0
    for out1,label in zip(outs,flowl0):
        mean_RMSE+=RMSE(out1.detach(),label.detach().cuda())
    mean_RMSE=mean_RMSE/7
    vis['mean_RMSE'] =mean_RMSE
    return total_loss, vis,rmse_1

def main():
    TrainImgLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                                 drop_last=True,pin_memory=True)


    start_full_time = time.time()
    start_epoch = 1 if args.casc_resume is None else int(re.findall('(\d+)', args.casc_resume)[0]) + 1

    viz_1 = Visdom()

    viz_1.line([[0., 0., 0.]], [start_epoch], win='train',
               opts=dict(title='train_loss&test_loss', legend=['mean_rmse', 'first_rmse', 'learn_rate*1e5']))
    for epoch in range(start_epoch, args.epochs + 1):
        total_train_loss = 0
        total_train_rmse = 0
        total_first_rmse=0
        # training loop
        total_iters = 0
        for batch_idx, (imgL_crop, imgR_crop,flow0) in enumerate(TrainImgLoader):
            start_time = time.time()
            loss, vis,rmse_1 = train(imgL_crop, imgR_crop, flow0)
            if (total_iters) % 50 == 0:
                print('Epoch %d Iter %d/%d mean_RMSE = %.3f   first_RMSE = %.3f ,  , time = %.2f， learn rate*1e5=%.8f' % (epoch,
                                                                                             batch_idx,len(TrainImgLoader),
                                                                                             vis['mean_RMSE'],
                                                                                            rmse_1,
                                                                                             time.time() - start_time,
                                                                                               1e5*scheduler.optimizer.param_groups[0]['lr']  ))
            total_train_loss += loss
            total_first_rmse+=rmse_1
            total_train_rmse += vis['mean_RMSE']
            learn_rate = scheduler.optimizer.param_groups[0]['lr']
            total_iters += 1
        #scheduler.step(epoch)

        print('Epoch {}, Mean train_rmse = {:.4f}  Mean_first_rmse = {:.4f}'.format(epoch, total_train_rmse / total_iters, total_first_rmse/total_iters ))
        learn_rate = scheduler.optimizer.param_groups[0]['lr']
        learn_show = learn_rate * 100000
        mean_rmse=total_train_rmse / total_iters
        first_rms=total_first_rmse/total_iters
        scheduler.step(mean_rmse)
        if epoch %3==0:
            casc_filename=args.savemodel + '/' + args.casc_logname + '/finetune_' + str(epoch) + '.tar'
            casc_save_dict=casc_model.state_dict()
            casc_save_dict=collections.OrderedDict(
                {k: v for k, v in casc_save_dict.items() if ('flow_reg' not in k or 'conv1' in k) and ('grid' not in k)})
            torch.save(
                {'epoch': epoch, 'state_dict': casc_save_dict, 'mean_rmse_loss': mean_rmse,'vis':vis },
                casc_filename)

        viz_1.line([[mean_rmse.detach().cpu().numpy(), first_rms.detach().cpu().numpy(), learn_show]], [epoch], win='train', update='append')


    torch.cuda.empty_cache()


    print('full finetune time = %.2f HR' % ((time.time() - start_full_time) / 3600))


if __name__ == '__main__':
    main()
