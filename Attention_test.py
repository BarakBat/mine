from Attention import Attention, MultiHeadAttention
import torch
def main():
    in_sen_len=16
    embedding_size=64
    heads=4
    batch_size=1
    d_model=64
    k = torch.rand(batch_size,in_sen_len,embedding_size)
    q = torch.rand(batch_size,in_sen_len,embedding_size)
    v = torch.rand(batch_size,in_sen_len,embedding_size)

    print(k,q,v)
    mha=MultiHeadAttention(in_sen_len,embedding_size,heads,d_model)
    mha = mha(k,q,v)


main()