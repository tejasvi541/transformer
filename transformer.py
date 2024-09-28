import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):
    
    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # alpha and bias are learnable parameters (1D tensor of shape (features,))
        self.alpha = nn.Parameter(torch.ones(features))  # (features,)
        self.bias = nn.Parameter(torch.zeros(features))  # (features,)

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1) - mean along hidden_size
        std = x.std(dim=-1, keepdim=True)    # (batch, seq_len, 1) - std along hidden_size
        # Normalize, then scale by alpha and shift by bias
        return self.alpha * (x - mean) / (std + self.eps) + self.bias  # (batch, seq_len, hidden_size)

class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # (d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # (d_ff, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model) -> after Linear(1): (batch, seq_len, d_ff)
        # After ReLU activation -> Dropout -> Linear(2): (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)  # (vocab_size, d_model)

    def forward(self, x):
        # x: (batch, seq_len) -> (batch, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)  # Scale embeddings

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Positional encoding: (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2)
        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # (seq_len, d_model//2)
        pe[:, 1::2] = torch.cos(position * div_term)  # (seq_len, d_model//2)
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model) -> Add positional encoding
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)  # (batch, seq_len, d_model)
        return self.dropout(x)

class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)  # (features,)

    def forward(self, x, sublayer):
        # Apply normalization -> sublayer -> dropout, and add the original input (residual)
        return x + self.dropout(sublayer(self.norm(x)))  # (batch, seq_len, features)

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h  # Number of heads
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h  # Dim per head
        # Linear transformations for queries, keys, values, and output (all d_model x d_model)
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # (d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # (d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # (d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # (d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]
        # Attention score: (batch, h, seq_len, d_k) @ (batch, h, d_k, seq_len) -> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) @ (batch, h, seq_len, d_k) -> (batch, h, seq_len, d_k)
        return attention_scores @ value, attention_scores

    def forward(self, q, k, v, mask):
        # Linear projections: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        query = self.w_q(q)  # (batch, seq_len, d_model)
        key = self.w_k(k)    # (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model)

        # Reshape: (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)  # (batch, h, seq_len, d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)  # (batch, h, seq_len, d_k)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)  # (batch, h, seq_len, d_k)

        # Apply attention: (batch, h, seq_len, d_k)
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Concatenate heads and reshape: (batch, h, seq_len, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)  # (batch, seq_len, d_model)

        # Apply final linear layer (output projection)
        return self.w_o(x)  # (batch, seq_len, d_model)

class EncoderBlock(nn.Module):
    # This block performs multi-head self-attention followed by a feed-forward layer with residual connections.
    # Inputs:
    # - features: Dimension of the input (d_model).
    # - self_attention_block: Multi-head attention module.
    # - feed_forward_block: Feed-forward block.
    # - dropout: Dropout probability.

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # Residual connections around attention and feed-forward blocks
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # Apply multi-head self-attention with residual connection
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))  # (batch, seq_len, d_model)
        # Apply feed-forward block with residual connection
        x = self.residual_connections[1](x, self.feed_forward_block)  # (batch, seq_len, d_model)
        return x

class Encoder(nn.Module):
    # Encoder module composed of multiple encoder blocks.
    # Inputs:
    # - features: Dimension of the input (d_model).
    # - layers: List of encoder blocks.

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        # Pass input through each encoder block
        for layer in self.layers:
            x = layer(x, mask)  # (batch, seq_len, d_model)
        return self.norm(x)  # Apply layer normalization after all layers

class DecoderBlock(nn.Module):
    # Decoder block that performs masked self-attention, cross-attention, and feed-forward processing with residual connections.
    # Inputs:
    # - features: Dimension of the input (d_model).
    # - self_attention_block: Multi-head self-attention module.
    # - cross_attention_block: Multi-head cross-attention module.
    # - feed_forward_block: Feed-forward block.
    # - dropout: Dropout probability.

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        # Residual connections around self-attention, cross-attention, and feed-forward blocks
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Apply masked multi-head self-attention with residual connection
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))  # (batch, seq_len, d_model)
        # Apply multi-head cross-attention with residual connection
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))  # (batch, seq_len, d_model)
        # Apply feed-forward block with residual connection
        x = self.residual_connections[2](x, self.feed_forward_block)  # (batch, seq_len, d_model)
        return x

class Decoder(nn.Module):
    # Decoder composed of multiple decoder blocks.
    # Inputs:
    # - features: Dimension of the input (d_model).
    # - layers: List of decoder blocks.

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Pass input through each decoder block
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)  # (batch, seq_len, d_model)
        return self.norm(x)  # Apply layer normalization after all layers

class ProjectionLayer(nn.Module):
    # Linear layer to project the final decoder output to the vocabulary size.
    # Inputs:
    # - d_model: Dimension of the model's output.
    # - vocab_size: Size of the target vocabulary.

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # Project (batch, seq_len, d_model) to (batch, seq_len, vocab_size)
        return self.proj(x)

class Transformer(nn.Module):
    # Full transformer model consisting of the encoder, decoder, and embedding layers.
    # Inputs:
    # - encoder: Encoder object.
    # - decoder: Decoder object.
    # - src_embed: Source embedding layer.
    # - tgt_embed: Target embedding layer.
    # - src_pos: Source positional encoding.
    # - tgt_pos: Target positional encoding.
    # - projection_layer: Linear projection layer for the output.

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # Embed and encode the source input.
        # (batch, seq_len) --> (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # Embed and decode the target input, considering the encoder output.
        # (batch, seq_len) --> (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # Project decoder output to the vocabulary size.
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create a transformer model with the specified hyperparameters.
    # Inputs:
    # - src_vocab_size: Source vocabulary size.
    # - tgt_vocab_size: Target vocabulary size.
    # - src_seq_len: Maximum length of source sequences.
    # - tgt_seq_len: Maximum length of target sequences.
    # - d_model: Dimension of the model (hidden layer size).
    # - N: Number of encoder and decoder layers.
    # - h: Number of attention heads.
    # - dropout: Dropout probability.
    # - d_ff: Dimension of the feed-forward layer.

    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters using Xavier initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

