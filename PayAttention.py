import torch
from torch import nn
from TimeWizard import Wiz
from Layers import PositionalEncoder, Encoder,Encoder_no_ffn



class PayAttention(nn.Module):
    def __init__(self, feature_size, frame_size,attention_batch,frames4attention,attention_arch,cnn_arch):
        super(PayAttention, self).__init__()
        self.feature_size    = feature_size
        self.frame_size      = frame_size
        self.attention_batch      = attention_batch
        self.pos_enc         = PositionalEncoder(feature_size,cnn_arch,frames4attention)
        self.frames4attention = frames4attention
        self.attention_arch  = attention_arch
        self.cnn_arch        = cnn_arch
        if attention_arch == "dec":
            self.new_frames4attention=int(frames4attention/2)
            self.SelfAttention1        = Encoder_no_ffn(self.new_frames4attention,4,feature_size,1200,attention_batch)
            self.SelfAttention2        = Encoder_no_ffn(self.new_frames4attention,4,feature_size,1200,attention_batch)
            self.Attention             = Wiz(self.new_frames4attention,4,feature_size,1200,attention_batch)
        else:
            self.new_frames4attention  = frames4attention
            self.SelfAttention1        = Encoder(frames4attention, 4, feature_size, 1200, attention_batch)
            self.SelfAttention2        = Encoder(frames4attention, 4, feature_size, 1200, attention_batch)
    def forward(self,x):
        x = torch.squeeze(x)
        x = self.pos_enc(x, self.attention_batch)
        if self.cnn_arch != "mfnet":
            x = x.view(self.attention_batch, self.frames4attention)

        if self.attention_arch == "dec":
            input = torch.split(x,(self.new_frames4attention,self.new_frames4attention),1)
            input1=input[0]
            input2=input[1]
            out1 = self.SelfAttention1(input1)
            out2 = self.SelfAttention1(input2)
            out = self.Attention(out1,out2)
        else:
            out = self.SelfAttention2(x)

        return out