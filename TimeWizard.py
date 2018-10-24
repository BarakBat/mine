from Layers import MultiHeadAttention,FeedForwardNetwork, NormAdd
from torch import nn

class Wiz(nn.Module):
    def __init__(self,frames_sequence,heads,feature_size,hidden_size,batch_size):
        super(Wiz, self).__init__()
        self.weights4firstpart=0.5
        self.time_attn = MultiHeadAttention(frames_sequence,feature_size,heads,batch_size)
        self.normadd=NormAdd([frames_sequence,feature_size])
        self.FFN= FeedForwardNetwork(feature_size,hidden_size)

    def forward(self, input1, input2, non_pad_mask=None, slf_attn_mask=None, dec_time_attn_mask=None):
        '''input1-from the 1st part, input2 -from 2nd part'''

        dec_output, dec_time_attn = self.time_attn(input1, input1, input2, mask=dec_time_attn_mask)

       # dec_output *= non_pad_mask
        residual = self.weights4firstpart*input1 +(1-self.weights4firstpart)*input2
        residual = self.normadd(dec_output,residual)
        dec_output = self.FFN(residual)
        dec_output = self.normadd(dec_output,residual)

        #dec_output *= non_pad_mask

        return dec_output