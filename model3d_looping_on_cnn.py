import torch
from torch import nn
from net2 import ResNet
#from models import resnet, pre_act_resnet, wide_resnet, resnext, densenet
import mobilenet
from mfnet import MFNET_3D
import resnet
#from resnet import ResNet
from models import resnet as resnet3d

from PayAttention import PayAttention
def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters




class BNet(nn.Module):
    def __init__(self,opt):
        super(BNet, self).__init__()
        #Defines
        self.number_gpu=opt.number_gpu
        self.cnn_arch = opt.cnn_arch
        self.feature_size = opt.feature_size
        self.frame_size = opt.frame_size
        self.batch_size = int(opt.batch_size/opt.number_gpu)
        self.frames_sequence = opt.frames_sequence
        self.split_input = opt.split_input
        self.no_cuda = opt.no_cuda
        self.frames4attention = opt.frames4attention
        self.batch_load = int(self.batch_size / self.split_input) # batch in each iteration
        self.attention_batch = int(self.batch_size / self.frames4attention)

        ###feature extractor###
        if self.cnn_arch == '2D':
            self.feature_extrator = resnet.resnet50(feature_size=opt.feature_size)
        elif self.cnn_arch == '3D':
            self.feature_extrator = resnet3d.resnet34(feature_size=opt.feature_size, frame_size=opt.frame_size,frames_sequence=opt.frames_sequence)
        elif self.cnn_arch == 'mfnet':
            self.feature_extrator =MFNET_3D(opt.feature_size)
        ###change dimansion###
        if opt.cnn_arch == 'mfnet':
            self.dim_ds           = nn.Linear(768, opt.feature_size_ds)
        else:
            self.dim_ds           = nn.Linear(opt.feature_size, opt.feature_size_ds)
        for p in self.dim_ds.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        ###attention####
        if self.cnn_arch == "mfnet":
            feature_size_ds = 400
        else:
            feature_size_ds = opt.feature_size_ds
        self.attention = PayAttention(feature_size_ds, opt.frame_size, self.attention_batch, opt.frames4attention,opt.attention_arch,opt.cnn_arch)
        for p in self.attention.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        ####fc3####
        if opt.attention_arch == "dec":
             self.fc3 = nn.Linear(int(opt.frames4attention/2), opt.n_classes)
        else:
            self.fc3 = nn.Linear(opt.frames4attention, opt.n_classes)
        for p in self.fc3.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        ####fc4####
        self.fc4 = nn.Linear(feature_size_ds, 1)
        for p in self.fc4.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x,frames_sequence):

        x_list=[]
        input2attention = []
        x_list = torch.split(x,self.batch_load,0)
        for i in range(self.split_input):
            if not self.no_cuda:
                item = x_list[i].to('cuda')
            else:
                item = x_list[i]
            if self.cnn_arch == '3D':
                item = item.view(self.batch_load, 3, self.frames_sequence, self.frame_size, self.frame_size)
            elif self.cnn_arch == '2D':
                item = item.view(self.batch_load * self.frames_sequence, 3, self.frame_size, self.frame_size)
            elif self.cnn_arch == 'mfnet':
                item = item.view(self.batch_load, 3, self.frames_sequence, self.frame_size, self.frame_size)
            with torch.no_grad():
                input2attention.append(self.feature_extrator(item))
        z = torch.cat(input2attention,dim=0)
        if self.cnn_arch != 'mfnet':
            z = self.dim_ds(z)
        else:
            z = z.view(self.attention_batch,self.frames4attention,z.shape[1])
        z = self.attention(z)
        z = z.transpose(1,2)
        z = self.fc3(z)
        z = z.transpose(1,2)
        z = self.fc4(z)


        return z

def model(opt):
    model = BNet(opt)

    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)

    return model, model.parameters()

