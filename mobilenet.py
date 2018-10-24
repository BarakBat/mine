import torch
from torch import nn

from Layers import pos_enc_table, PositionalEncoder, Encoder
class Net(nn.Module):
    def __init__(self,num_classes, feature_size, frame_size, frames_sequence,batch_size):
        super(Net, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(    32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1, 1),

        )
        self.feature_size    = feature_size
        self.frame_size      = frame_size
        self.batch_size      = batch_size
        self.frames_sequence = frames_sequence
        self.fc1             = nn.Linear(7, 256)
        self.fc2             = nn.Linear(7, 1)
        self.pool            = nn.AvgPool1d(7)
        self.pos_enc         = PositionalEncoder(feature_size,frames_sequence)
        self.Learner1        = Encoder(256,frames_sequence,4,256,512,batch_size)
        self.Learner2         = Encoder(256,frames_sequence,4,256,512,batch_size)
        self.fc3             = nn.Linear(frames_sequence, num_classes)
        self.fc4             = nn.Linear(feature_size, 1)
        self.softmax         = nn.Softmax()


    def forward(self, x,frames_sequence):
        x = torch.squeeze(x)
        # batch_size, channels , frame_seq,size,size -> channels, batch_size*frame_seq,size,size
        x = x.view(self.batch_size*self.frames_sequence, 3, self.frame_size, self.frame_size)
        #x = x.permute(1,0,2,3)
        x = self.model(x)
        x = x.view(self.batch_size*self.frames_sequence, 7, 7)
        x = self.fc1(x)
        x = x.permute(0,2,1)
        #x = torch.transpose(x, 0, 1)
        x = self.fc2(x)
        x = torch.squeeze(x)
        x = self.pos_enc(x,self.batch_size)
        x=x.view(self.batch_size,self.frames_sequence,self.feature_size)
        x = self.Learner1(x)
        x = self.Learner2(x)
        x = x.transpose(1,2)
        x = self.fc3(x)
        x = x.transpose(1,2)
        x = self.fc4(x)
        #x = self.softmax(torch.squeeze(x))
        x = self.softmax(x)
        return x

""""
if __name__ == "__main__":
    frames_sequence=2
    feature_Size=256
    x=torch.rand(frames_sequence, 3, 224, 224)
    model=Net(frames_sequence, feature_Size)
    output=model(x,frames_sequence)
    print(str(model))
    #summary(model, input_size=(3, 224, 224))"""