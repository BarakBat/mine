import torch
from torch import nn
from TimeWizard import Wiz
from Layers import PositionalEncoder, Encoder,Encoder_no_ffn



class PayAttention(nn.Module):
    def __init__(self, feature_size, frame_size,batch_size,frames_sequence,attention_arch):
        super(PayAttention, self).__init__()
        self.feature_size    = feature_size
        self.frame_size      = frame_size
        self.batch_size      = batch_size
        self.pos_enc         = PositionalEncoder(feature_size,frames_sequence)
        self.frames_sequence = frames_sequence
        self.attention_arch  = attention_arch
        if attention_arch == "dec":
            self.new_frames_sequence=int(frames_sequence/2)
            self.SelfAttention1        = Encoder_no_ffn(self.new_frames_sequence,4,feature_size,1200,batch_size)
            self.SelfAttention2        = Encoder_no_ffn(self.new_frames_sequence,4,feature_size,1200,batch_size)
            self.Attention             = Wiz(self.new_frames_sequence,4,feature_size,1200,batch_size)
        else:
            self.new_frames_sequence = frames_sequence
            self.SelfAttention1        = Encoder(frames_sequence,4,feature_size,1200,batch_size)
            self.SelfAttention2        = Encoder(frames_sequence,4,feature_size,1200,batch_size)
    def forward(self,x):
        x = torch.squeeze(x)
        x = self.pos_enc(x, self.batch_size)
        x = x.view(self.batch_size, self.frames_sequence, self.feature_size)
        if self.attention_arch == "dec":
            input = torch.split(x,(self.new_frames_sequence,self.new_frames_sequence),1)
            input1=input[0]
            input2=input[1]
            out1 = self.SelfAttention1(input1)
            out2 = self.SelfAttention1(input2)
            out = self.Attention(out1,out2)
        else:
            x = self.SelfAttention2(x)

        return out