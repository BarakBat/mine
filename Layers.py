import torch
from torch import nn
import numpy as np
import math
from torch.autograd import Variable

#Defines
PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

class NormAdd(nn.Module):
    def __init__(self,frames4attention):
        super(NormAdd,self).__init__()
        self.norm = nn.LayerNorm(frames4attention)

    def forward(self,input,residual):
        output = self.norm(input + residual)
        return output

###need to understand and change
class FeedForwardNetwork(nn.Module):
    def __init__(self, feature_size, d_ff=2048, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(feature_size, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, feature_size)
        self.relu=nn.ReLU()

    def forward(self, x):
        x = self.dropout(self. relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Attention(nn.Module):
    def __init__(self,feature_size,dropout):
        super(Attention,self).__init__()
        self.scaler=math.sqrt(feature_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax=nn.Softmax2d()
    def forward(self,q,k,v,feature_size,mask=None):
        res=torch.matmul(q,k.transpose(-2,-1))/self.scaler

      #  if mask is not None:
      #      res = res.masked_fill(mask, -np.inf)

        res = self.softmax(res)
        res = self.dropout(res)
        attention = res
        res = torch.matmul(res,v)

        return res, attention #they also returned attn look at their code

class MultiHeadAttention(nn.Module):
    def __init__(self,frames4attention,feature_size,heads,attention_batch,dropout=0.12):
        super(MultiHeadAttention,self).__init__()
        self.frames4attention = frames4attention
        self.heads = heads
        self.dk = int(feature_size/heads)
        self.dv = int(feature_size/heads)
        self.dropout = nn.Dropout(dropout)
        self.attention_batch = attention_batch
        self.q_fc = nn.Linear(feature_size, feature_size)
        self.k_fc = nn.Linear(feature_size, feature_size)
        self.v_fc = nn.Linear(feature_size, feature_size)


        nn.init.normal_(self.q_fc.weight, mean=0, std=np.sqrt(2.0 / (feature_size + self.dk)))
        nn.init.normal_(self.k_fc.weight, mean=0, std=np.sqrt(2.0 / (feature_size + self.dv)))
        nn.init.normal_(self.v_fc.weight, mean=0, std=np.sqrt(2.0 / (feature_size + self.dk)))

        self.attention = Attention(feature_size,dropout)
        self.linear2 = nn.Linear(feature_size, feature_size)
        self.layer_norm = nn.LayerNorm(feature_size)

        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self,k,q,v,mask=None):


        q = self.q_fc(torch.squeeze(q)).view(self.attention_batch,self.frames4attention ,self.heads, self.dk)
        k = self.k_fc(torch.squeeze(k)).view(self.attention_batch,self.frames4attention ,self.heads, self.dk)
        v = self.v_fc(torch.squeeze(v)).view(self.attention_batch,self.frames4attention ,self.heads, self.dk)

        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

       # if mask is not None:
       #    mask = mask.repeat(self.heads, 1, 1)  # (n*b) x .. x ..
        res, attention = self.attention(q, k, v,self.dk*self.heads, mask=mask)
        #res = res.view(self.heads, attention_batch, self.frames4attention, self.dv)
        res = res.permute(1, 2, 0, 3).contiguous().view(self.attention_batch, self.frames4attention, -1)  # b x lq x (n*dv)
        res = self.linear2(res)
        res = self.dropout(res)

        return res, attention

        return output


class PositionalEncoder(nn.Module):
    def __init__(self, feature_size,cnn_arch, max_seq_len=80):
        super().__init__()
        self.feature_size = feature_size
        self.cnn_arch =cnn_arch
        # create constant 'pe' matrix with values dependant on
        # pos and i
        if cnn_arch == "mfnet":
            feature_size = 400
        pe = torch.zeros(max_seq_len, feature_size)
        for pos in range(max_seq_len):
            for i in range(0, feature_size, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / feature_size)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / feature_size)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, attention_batch):
        # make embeddings relatively larger
        x = x * math.sqrt(self.feature_size)
        # add constant to embedding
        seq_len = x.size(-1)
        pos = Variable(self.pe[:, :seq_len],requires_grad=False)
        if self.cnn_arch == "2D":
            pos = torch.squeeze(pos)
            pos = pos.repeat(attention_batch,1)
        else:
            pos=pos.repeat(x.shape[0], 1, 1)
        x = x + pos
        return x
#TODO: need to finish check + change, 4 functions
def pos_enc_table(frames4attention, feature_size, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / feature_size)

    def get_posi_angle_vec(position):
        #Each position gets embedding_size numbers
        return [cal_angle(position, i) for i in range(feature_size)]

    #For loop for each position in the sentence
    for pos_i in range(frames4attention):
        sinusoid_table = np.array([get_posi_angle_vec(pos_i)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)
def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(PAD).type(torch.float).unsqueeze(-1)
def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask
def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask



class Encoder(nn.Module):
    def __init__(self,frames4attention,heads,feature_size,hidden_size,attention_batch):
        super(Encoder,self).__init__()
        self.mha=MultiHeadAttention(frames4attention,feature_size,heads,attention_batch)
        self.normadd=NormAdd([frames4attention,feature_size])
        self.FFN= FeedForwardNetwork(feature_size,hidden_size)

    def forward(self, enc_in):

        output, atten=self.mha(enc_in,enc_in,enc_in)
        residual = self.normadd(output,enc_in)
        output = self.FFN(residual)
        output = self.normadd(output,residual)
        return output

class Encoder_no_ffn(nn.Module):
    def __init__(self,frames4attention,heads,feature_size,hidden_size,attention_batch):
        super(Encoder_no_ffn,self).__init__()
        self.mha=MultiHeadAttention(frames4attention,feature_size,heads,attention_batch)
        self.normadd=NormAdd([frames4attention,feature_size])

    def forward(self, enc_in):

        output, atten=self.mha(enc_in,enc_in,enc_in)
        residual = self.normadd(output,enc_in)

        return output


class Encoder_Stack(nn.Module):
    def __init__(self,frames4attention,feature_size,enc_layers,heads,embedding_size,hidden_size,attention_batch):
        super().__init__()

        self.layer_stack = nn.ModuleList([
            Encoder(feature_size, frames4attention, heads,embedding_size,hidden_size)
            for _ in range(enc_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)


        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


