import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape(seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector of length(seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # calculated in log space
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        # Apply the sin to even  and cos to odd positions
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:,:x.shape[1], :]).requires_grad(False)
        return self.dropout(x)
    
class LayerNormalisation(nn.Module):
    def __init__(self, eps:float = 10**-6) -> None :
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) 
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x-mean)/(std+self.eps)+self.bias

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        #(batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model:int, num_heads:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model should be divisible by num_heads"

        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout:nn.Dropout):
        d_k = query.shape[-1]
        # Calculate the Attention scores using the scaled dot product
        # (batch, num_heads, seq_len, d_k) x (batch, num_heads, d_k, seq_len) -> (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(d_k)
        # Apply the mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # Apply the softmax
        scores = torch.softmax(scores, dim = -1) # (batch, num_heads, seq_len, seq_len) -> (batch, num_heads, seq_len, seq_len)
        if dropout is not None:
            scores = dropout(scores)
        # Multiply the scores with the value
        # (batch, num_heads, seq_len, seq_len) x (batch, num_heads, seq_len, d_k) -> (batch, num_heads, seq_len, d_k)
        x = torch.matmul(scores, value)
        return x, scores


    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model)->(batch, seq_len, d_model)
        key = self.w_k(k)   # (batch, seq_len, d_model)->(batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model)->(batch, seq_len, d_model)

        # Split the query, key and value into num_heads
        # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        # Calculate the Attention scores using the scaled dot product
        
        x, self.attention_scores =  MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # Concatenate the num_heads
        # (batch, num_heads, seq_len, d_k) -> () ->(batch, seq_len, num_heads*d_k)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads*self.d_k)

        # Apply the final linear layer
        x = self.w_o(x)
        return x


print("No Errorrrrrr")