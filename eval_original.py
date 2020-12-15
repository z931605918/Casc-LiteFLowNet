from torchsummary import summary
import numpy as np
import cv2 as cv
from torch.utils.data import Dataset
from glob import glob
import os
import torch
from  torch.autograd import Variable
from utils.multiscaleloss import realEPE,RMSE
from tqdm import tqdm
from utils.dataloader import MyDataset
import PIL.Image as image
import time
from utils.augmentations import Augmentation, Basetransform
from lite_flownet import liteflownet as lite
from run_en import lietflownet as lite_en
folder='D:\desktop\连续流场数据集\测试数据集\空间\\N30_S40'  #测试数据集
save_name='D:\\desktop\\TCN-flownet\\程序\\Casc-LiteFlownet\\test_outcome\\lite_en_outcome {}.tar'.format(folder[-8:])
model = lite(
    'D:\desktop\Lite-Flownet-master\logname_old\\finetune_1272.tar')  # 模型储存于D:\desktop\Lite-Flownet-master\logname_en
#model=lite_en('D:\desktop\Lite-Flownet-master\logname_en_two\\finetune_354.tar')
model.cuda().eval()
num_img=8
def load_data(path):
    all_image_pathes = os.listdir(path)
    image_vel_pathes = []
    for i in all_image_pathes:
        image_vel_path = os.path.join(path, i)
        image_vel_pathes.append(image_vel_path)
    vel_pathes = image_vel_pathes
    image_pathes = image_vel_pathes
    vel_pathes = list(filter(lambda x: x.endswith('.flo'), vel_pathes))
    image_pathes = list(filter(lambda x: x.endswith('.bmp'), image_pathes))
    img_group=[]
    vel_group=[]
    for i in range(int(len(image_pathes)/num_img)):
        img_mini_group = []
        for j in range(num_img):
            img_mini_group.append(image_pathes[j+num_img*i])
        img_group.append(img_mini_group)
    for i in range(int(len(vel_pathes)/(num_img-1))):
        vel_mini_group = []
        for j in range(num_img-1):
            vel_mini_group.append(vel_pathes[j+(num_img-1)*i])
        vel_group.append(vel_mini_group)
    return img_group,vel_group
def read_flo_file(filename):
    """
    Read from Middlebury .flo file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        #print("Reading %d x %d flow file in .flo format" % (h, w))
        flow = np.fromfile(f, np.float32, count=2 * w[0] * h[0])
        # reshape data into 3D array (columns, rows, channels)
        flow = np.resize(flow, ( h[0], w[0] ,2))
        #flow=flow.transpose(2,0,1)
        return flow
def pre_trans(tensor):
    array=np.asarray(tensor)
    img=cv.cvtColor(array,cv.COLOR_RGB2BGR)
    return img
class Or_Dataset(Dataset):
    def __init__(self, path,  transform):
        self.img_group, self.vel_group = load_data(path)
        self.transform = transform
    def __getitem__(self, i):
        imgs1=[]
        imgs2=[]
        flos=[]
        for j in range(num_img-1):
            if self.transform is not None:
                img1 = image.open(self.img_group[i][j]).convert('RGB')
                img2 = image.open(self.img_group[i][j + 1]).convert('RGB')
                flo=read_flo_file(self.vel_group[i][j])
                img1, img2, flo = pre_trans(img1), pre_trans(img2), flo
                img1, img2, flo = self.transform(img1, img2, flo)
                imgs1.append(img1)
                imgs2.append(img2)
                flos.append(flo)
        return imgs1, imgs2, flos
    def __len__(self):
        return len(self.img_group)

def use_model(imgL, imgR, flowl0,model):
    outs=[]
    num=1
    mean_RMSE=0
    for I1, I2 in zip(imgL[0:], imgR[0:]):
        I1, I2 = I1.cuda(), I2.cuda()
        output = model((I1, I2))
        outs.append(output)
        num += 1
    vis = {}
    vis['output1'] = outs[0].detach().cpu().numpy()

    for out1, label in zip(outs, flowl0):
        mean_RMSE += RMSE(out1.cpu(), label.cpu())
    mean_RMSE = mean_RMSE / (num_img-1)
    return vis, mean_RMSE

def main():
    with torch.no_grad():
        casc_dataset = Or_Dataset(folder,transform=Basetransform(size=256,mean=(128)))   #在这改变数据集 和图片尺寸
        CascLoader = torch.utils.data.DataLoader(casc_dataset, batch_size=1, shuffle=False, num_workers=2,
                                                 drop_last=True, pin_memory=True)
        casc_dataset.__len__()
        start_full_time = time.time()
        casc_vis={}
        losses=0
        loss_list=[]

        for batch_idx, (imgL_crop, imgR_crop,flow0) in enumerate(CascLoader):
            start_time = time.time()
            vis,mean_RMSE=use_model(imgL_crop,imgR_crop, flow0, model)
            casc_vis['group {}'.format(batch_idx)]=vis
            losses+=mean_RMSE
            loss_list.append(mean_RMSE)
            total_batch=batch_idx+1
            end_time=time.time()
            cost_time=end_time-start_time
            print('mean_RMSE {:.4f}  , time {:.4f}'.format(mean_RMSE,cost_time))

        end_full_time=time.time()
        total_cost_time=end_full_time-start_full_time
        mean_RMSE_loss=losses/total_batch
        print(' total mean loss {:.4f} ,total cost time {:.4f} '.format(mean_RMSE_loss,total_cost_time))
        torch.save({'cost_time':total_cost_time , 'mean_loss':mean_RMSE_loss , 'vis':vis ,'loss_list':loss_list
        },save_name
        )
if __name__=='__main__':
    main()



















