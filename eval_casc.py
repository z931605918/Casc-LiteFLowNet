import torch
import numpy as np
import torch.nn as nn
import torchvision
import time
import PIL.Image as image
import cv2 as cv
import matplotlib.pyplot as plt
from lite_flownet import liteflownet as flownet_or
from lite_flownet_en import lietflownet as flownet_en
from casc_flownet_4layer import casc_liteflownet as casc_4
from casc_flownet_en_4layer import casc_liteflownet as casc_en_4
from casc_flownet_4layer_relu import casc_liteflownet as casc_4_relu
from casc_flownet_en_4layer_relu import casc_liteflownet as casc_en_4layer_relu
from casc_flownet_5layer import casc_liteflownet as casc_5layer
from casc_flownet_3layer import casc_liteflownet as casc_3layer
from casc_flownet_5layer_relu import casc_liteflownet as casc_5layer_relu
from torch.utils.data import Dataset
import os
from utils.augmentations import Augmentation, Basetransform
from utils.multiscaleloss import MultiscaleLoss, realEPE, RMSE
#casc_model=casc_3layer('D:\desktop\TCN-flownet\程序\Casc-LiteFlownet\logname_casc_three_layer\\finetune_471.tar')
model=flownet_or('D:\desktop\Lite-Flownet-master\logname_old\\finetune_1272.tar')
#model=flownet_en('D:\\desktop\\TCN-flownet\\程序\\Casc-LiteFlownet\\logname_en\\finetune_336.tar')  #加载基础模型
casc_model=casc_4('D:\desktop\TCN-flownet\程序\Casc-LiteFlownet\logname_casc_four_layer\\finetune_255.tar')
casc_model_2=casc_4_relu('D:\desktop\TCN-flownet\程序\Casc-LiteFlownet\logname_casc_four_layer_relu\\finetune_300.tar')
#casc_model=casc_5layer('D:\desktop\TCN-flownet\程序\Casc-LiteFlownet\logname_casc_five_layer\\finetune_405.tar')
#casc_model=casc_5layer_relu('D:\desktop\TCN-flownet\程序\Casc-LiteFlownet\logname_casc_five_layer_relu\\finetune_1500.tar')

#casc_model=casc_en_4('D:\desktop\TCN-flownet\程序\Casc-LiteFlownet\logname_casc_en_four_layer\\finetune_312.tar')
#casc_model=casc_en_4layer_relu('D:\desktop\TCN-flownet\程序\Casc-LiteFlownet\logname_casc_en_four_layer_relu\\finetune_410.tar')
#folder='Ln_DNS30'    #改变文件夹
folder='中间层\\L00_S50'
#folder='Ln_DNS10'
#folder='Rd_Uni15'
save_name='D:\\desktop\\TCN-flownet\\程序\\Casc-LiteFlownet\\test_outcome\\casc1_outcome{}.tar'.format(folder)
#casc_model=casc_2('D:\\desktop\\TCN-flownet\\程序\\Casc-LiteFlownet\\logname_casc_en_re\\finetune_363.tar')  #加载串联模型
casc_model.cuda().eval()

model.cuda().eval()
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
    for i in range(int(len(image_pathes)/8)):
        img_mini_group = []
        for j in range(8):
            img_mini_group.append(image_pathes[j+8*i])
        img_group.append(img_mini_group)
    for i in range(int(len(vel_pathes)/7)):
        vel_mini_group = []
        for j in range(7):
            vel_mini_group.append(vel_pathes[j+7*i])
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
    tensor=np.asarray(tensor)
    img=cv.cvtColor(tensor,cv.COLOR_RGB2BGR)
    return img
class Casc_Dataset(Dataset):
    def __init__(self, path,  transform):
        self.img_group, self.vel_group = load_data(path)
        self.transform = transform

    def __getitem__(self, i):
        imgs1=[]
        imgs2=[]
        flos=[]
        for j in range(7):
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
def gen_mag_out_or(out,label=None):
    mag_out=[]
    i=0
    if label is not None:
        while i <=6:
            if i%2==0 :
                mag_out.append(torch.sqrt((out[i][0][0]-label[0][0]/(2**(i/2))).pow(2)+(out[i][0][1]-label[0][1]/(2**(i/2))).pow(2)).cpu().numpy())
            i+=1

    else:
        while i <=6:
            if i%2==0 :
                mag_out.append(torch.sqrt(out[i][0][0].pow(2)+out[i][0][1].pow(2)).cpu().numpy())
            i+=1

    return mag_out
def gen_mag_out_casc_or(out,label=None):
    mag_out=[]
    i=0
    if label is not None:
        while i <= 3:
            out[i] = torch.nn.functional.interpolate(out[i], size=(256, 256), mode='bilinear', align_corners=False)
            if i!=3:
                mag_out.append(torch.sqrt((out[i][0][0]-label[0][0]/(2**(i-1))).pow(2)+(out[i][0][1]-label[0][1]/(2**(i-1))).pow(2)).cpu().numpy())
                i+=1
            else:
                mag_out.append(torch.sqrt((out[i][0][0]-label[0][0]/(2)).pow(2)+(out[i][0][1]-label[0][1]/(2)).pow(2)).cpu().numpy())
                i += 1
    else:
        while i <=3:
            out[i]=torch.nn.functional.interpolate(out[i],size=(256,256),mode='bilinear',align_corners=False)
            mag_out.append(torch.sqrt(out[i][0][0].pow(2)+out[i][0][1].pow(2)).cpu().numpy())
            i+=1

    return mag_out
def gen_mag_out_casc(out,label=None):
    mag_out=[]
    i=0
    if label is not None:
        while i <= 3:
            if i!=3:
                out[i] = torch.nn.functional.interpolate(out[i], size=(256, 256), mode='bilinear', align_corners=False)
                mag_out.append(torch.sqrt((out[i][0][0]-label[0][0]/(2**(i-1))).pow(2)+(out[i][0][1]-label[0][1]/(2**(i-1))).pow(2)).cpu().numpy())
                i+=1
            if i==3:
                out[i] = torch.nn.functional.interpolate(out[i], size=(256, 256), mode='bilinear', align_corners=False)
                mag_out.append(torch.sqrt(((out[i][0][0]-label[0][0]/(2**(i-1))).pow(2)+(out[i][0][1]-label[0][1]/(2**(i-1))).pow(2))).cpu().numpy())
                i+=1
    else:
        while i <=3:
            out[i]=torch.nn.functional.interpolate(out[i],size=(256,256),mode='bilinear',align_corners=False)
            mag_out.append(torch.sqrt(out[i][0][0].pow(2)+out[i][0][1].pow(2)).cpu().numpy())
            i+=1

    return mag_out
def plot_middle_layers(input1,input2,input3,label):
    #原始模型
    plt.subplot(4,3,1)
    plt.imshow(torch.sqrt(label[0][0][0].pow(2)+label[0][0][1].pow(2)).cpu().numpy(),cmap='RdYlBu_r',vmax=5)
    plt.colorbar(shrink=1)
    plt.subplot(4,3,4)
    plt.imshow(input1[0],cmap='RdYlBu_r',vmax=1)
    plt.colorbar(shrink=1)
    plt.subplot(4,3,7)
    plt.imshow(input1[1],cmap='RdYlBu_r',vmax=2)
    plt.colorbar(shrink=1)
    plt.subplot(4,3,10)
    plt.imshow(input1[2],cmap='RdYlBu_r',vmax=2)
    plt.colorbar(shrink=1)

 #casc
    plt.subplot(4,3,2)
    plt.imshow(torch.sqrt(label[1][0][0].pow(2)+label[1][0][1].pow(2)).cpu().numpy(),cmap='RdYlBu_r',vmax=5)
    plt.colorbar(shrink=1)
    plt.subplot(4,3,5)
    plt.imshow(input2[1],cmap='RdYlBu_r',vmax=1)
    plt.colorbar(shrink=1)
    plt.subplot(4,3,8)
    plt.imshow(input2[2],cmap='RdYlBu_r',vmax=2)
    plt.colorbar(shrink=1)
    plt.subplot(4,3,11)
    plt.imshow(input2[3],cmap='RdYlBu_r',vmax=2)
    plt.colorbar(shrink=1)

    #casc_relu
    plt.subplot(4,3,3)
    plt.imshow(torch.sqrt(label[2][0][0].pow(2)+label[2][0][1].pow(2)).cpu().numpy(),cmap='RdYlBu_r',vmax=5)
    plt.colorbar(shrink=1)
    plt.subplot(4,3,6)
    plt.imshow(input3[1],cmap='RdYlBu_r',vmax=1)
    plt.colorbar(shrink=1)
    plt.subplot(4,3,9)
    plt.imshow(input3[2],cmap='RdYlBu_r',vmax=2)
    plt.colorbar(shrink=1)
    plt.subplot(4,3,12)
    plt.imshow(input3[3],cmap='RdYlBu_r',vmax=2)
    plt.colorbar(shrink=1)






    plt.show()




def plot_interlayer(imgL,imgR,flowl0,model=model,casc_model=casc_model):
    Im1 = imgL[0].cuda()
    Im2 = imgR[0].cuda()
    flowl0[0] = flowl0[0].cuda()
    out_1 = model((Im1, Im2))[0]
    Im3=imgR[1].cuda()
    out_or=model((Im2,Im3))
    flowl0[1]=flowl0[1].cuda()
    out_casc=casc_model((Im2,Im3), out_1)
    out_casc_2=casc_model_2((Im2,Im3),out_1)
    mag_out_or = gen_mag_out_or(out_or,flowl0[1])
    mag_out_casc=gen_mag_out_casc_or(out_casc,flowl0[1])
    mag_out_casc_2=gen_mag_out_casc(out_casc_2,flowl0[1])
    outs=[out_or[0],out_casc[0],out_casc_2[0]]
    plot_middle_layers(mag_out_or,mag_out_casc,mag_out_casc_2,outs)















def use_casc(imgL, imgR, flowl0,model=model,casc_model=casc_model):
    #RMSEloss = torch.nn.MSELoss()

    Im1 = imgL[0].cuda()
    Im2 = imgR[0].cuda()
    flowl0[0] = flowl0[0].cuda()
    outs = [model((Im1, Im2))[0]]
    num = 1
    mean_RMSE = 0
    vises=[]
    for I1, I2, flo in zip(imgL[1:], imgR[1:], flowl0[1:]):
        I1, I2, flo = I1.cuda(), I2.cuda(), flo.cuda()
        last_out = outs[-1]
        last_out=last_out.view(1,2,256,256)
        output = casc_model((I1, I2), last_out)
        outs.append(output[0])
        num += 1
    vis = {}
    vis['output1'] = outs[0].detach().cpu().numpy()
    vis['output2'] = outs[1].detach().cpu().numpy()
    vis['output3'] = outs[2].detach().cpu().numpy()
    vis['output4'] = outs[3].detach().cpu().numpy()
    vis['output5'] = outs[4].detach().cpu().numpy()
    vis['output6'] = outs[5].detach().cpu().numpy()
    vis['output7'] = outs[6].detach().cpu().numpy()
    first_RMSE=RMSE(torch.unsqueeze(outs[0].cpu(),dim=0), flowl0 [0].cpu())
    for out1, label in zip(outs, flowl0):
        mean_RMSE += RMSE(torch.unsqueeze(out1.cpu(),dim=0), label.cpu())
    mean_RMSE = mean_RMSE / 7
    vises.append(vis)
    return vises, mean_RMSE ,first_RMSE

def main():
    with torch.no_grad():
        casc_dataset = Casc_Dataset('D:\desktop\连续流场数据集\测试数据集\\'+folder,transform=Basetransform(size=256,mean=(128)))
        CascLoader = torch.utils.data.DataLoader(casc_dataset, batch_size=1, shuffle=False, num_workers=1,
                                                 drop_last=True, pin_memory=True)
        model.eval()
        casc_model.eval()
        start_full_time = time.time()
        casc_vis={}
        losses=0
        first_losses=0
        loss_list=[]
        for batch_idx, (imgL_crop, imgR_crop,flow0) in enumerate(CascLoader):
            start_time = time.time()

            vis,mean_RMSE,first_RMSE=use_casc(imgL_crop,imgR_crop, flow0)

            casc_vis['group {}'.format(batch_idx)]=vis
            first_losses+=first_RMSE
            losses+=mean_RMSE
            loss_list.append(mean_RMSE)
            total_batch=batch_idx+1
            end_time=time.time()
            cost_time=end_time-start_time
            print('mean_RMSE {:.4f} , first_RMSE {:.4f} , time {:.4f}  i {}'.format(mean_RMSE,first_RMSE,cost_time,batch_idx+1))
        end_full_time=time.time()
        cost_time=end_full_time-start_full_time
        mean_loss=losses/(total_batch)
        first_mean_loss=first_losses/total_batch
        print('total_mean_RMSE={:.4f} , total_first_RMSE={:.4f}  , total time ={:.4f} '.format(mean_loss,first_mean_loss,cost_time))

        #torch.save({'cost_time':cost_time , 'mean_loss':mean_loss , 'vis':vis ,'loss_list':loss_list ,'first_RMSE':first_mean_loss
        #},save_name
        #)
#if __name__=='__main__':
  #  main()
def eval(model,casc_model):
    with torch.no_grad():
        model=model.cuda().eval()
        casc_dataset = Casc_Dataset('D:\desktop\连续流场数据集\测试数据集\\Rd_DNS15', transform=Basetransform(size=256, mean=(128)))
        casc_dataset_2 = Casc_Dataset('D:\desktop\连续流场数据集\测试数据集\\Rd_Uni15', transform=Basetransform(size=256, mean=(128)))
        CascLoader = torch.utils.data.DataLoader(casc_dataset, batch_size=1, shuffle=False, num_workers=1,
                                             drop_last=True, pin_memory=True)
        CascLoader_2 = torch.utils.data.DataLoader(casc_dataset_2, batch_size=1, shuffle=False, num_workers=1,
                                             drop_last=True, pin_memory=True)
        casc_model=casc_model.cuda().eval()
        losses=0
        losses_2=0
        for batch_idx, (imgL_crop, imgR_crop, flow0) in enumerate(CascLoader):
            vis, mean_RMSE, first_RMSE = use_casc(imgL_crop, imgR_crop, flow0,model,casc_model)
            losses += mean_RMSE
            total_batch = batch_idx
        total_rmse=losses/total_batch
        for batch_idx, (imgL_crop, imgR_crop, flow0) in enumerate(CascLoader_2):
            vis_2, mean_RMSE_2, first_RMSE_2 = use_casc(imgL_crop, imgR_crop, flow0, model, casc_model)
            losses_2 += mean_RMSE_2
            total_batch = batch_idx
        total_rmse_2=losses_2/total_batch
        return total_rmse.cpu().detach().numpy(),total_rmse_2.cpu().detach().numpy()

def plot_interlayer_out():
    with torch.no_grad():
        casc_dataset = Casc_Dataset('D:\desktop\连续流场数据集\测试数据集\\'+folder,transform=Basetransform(size=256,mean=(128)))
        CascLoader = torch.utils.data.DataLoader(casc_dataset, batch_size=1, shuffle=False,
                                                 drop_last=True, pin_memory=True)
        model.train()
        casc_model.train()
        casc_model_2.cuda().train()

        for batch_idx, (imgL_crop, imgR_crop, flowl0) in enumerate(CascLoader):
            plot_interlayer(imgL_crop, imgR_crop, flowl0)
plot_interlayer_out()
