import torch
from torch import nn
from TimeWizard import Wiz
from Layers import PositionalEncoder, Encoder,Encoder_no_ffn



class PayAttention2D(nn.Module):
    def __init__(self, feature_size, frame_size,batch_size,frames_sequence,attention_arch):
        super(PayAttention2D, self).__init__()
        self.feature_size    = feature_size
        self.frame_size      = frame_size
        self.batch_size      = batch_size
        self.frames_sequence = frames_sequence
        self.pos_enc         = PositionalEncoder(feature_size,"2D",frames_sequence)

        if attention_arch == "dec":
            self.SelfAttention1        = Encoder_no_ffn(frames_sequence[0:batch_size/2],4,feature_size,1200,batch_size)
            self.SelfAttention2        = Encoder_no_ffn(frames_sequence[batch_size/2+1:],4,feature_size,1200,batch_size)
            self.Attention             = Wiz(frames_sequence,4,feature_size,1200,batch_size)
        else:
            self.SelfAttention1        = Encoder(frames_sequence,4,feature_size,1200,batch_size)
            self.SelfAttention2        = Encoder(frames_sequence,4,feature_size,1200,batch_size)
    def forward(self,x):
        x = torch.squeeze(x)
        x = self.pos_enc(x, self.batch_size)
        x = x.view(self.batch_size, self.frames_sequence, self.feature_size)
        x = self.SelfAttention1(x)
        #x = self.SelfAttention2(x)

        return x