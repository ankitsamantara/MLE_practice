import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size:int, d_model:int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

    def forward(self, x):
        return self.embedding(x)*math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, max_seq_len:int=5000, dropout:float=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension so dimension -> (1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)

        # Register the positional encoding as a buffer so it is not a model parameter
        self.register_buffer('pe', pe)  

    def forward(self, x):
        # Add positional encoding to the input embeddings
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, d_model:int, eps:float=1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # Compute the mean and variance for layer normalization
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.alpha * x + self.bias

class FeedforwardBlock(nn.Module):
    def __init__(self, d_model:int, hidden_size:int, dropout:float=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model:int, h:int, dropout:float=0.1):
        super().__init__()
        self.d_model = d_model
        self.h = h

        assert self.d_model % self.h == 0, "d_model must be divisible by h"

        self.d_k = self.d_model // self.h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_score = (query @ key.transpose(-2,-1))/math.sqrt(d_k)

        if mask is not None:
            attention_score.masked_fill_(mask==0,-1e9)
        
        # Dimension -> (batch_size,seq_len,seq_len)
        attention_score = attention_score.softmax(dim = -1)

        if dropout is not None:
            attention_score = dropout(attention_score)
        
        return (attention_score@value), attention_score

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # Dimensions (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        key = self.w_k(k) # Dimensions (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        value = self.w_v(v) # Dimensions (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)

        # Dimensions (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2) 
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2) 
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2) 

        x, self.attention_score = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Dimension : (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1, self.h * self.d_k)

        # Dimension : (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        return (self.w_o(x))

class ResidualConnection(nn.Module):
    def __init__(self,dropout:float, d_model:int)->None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedforwardBlock, dropout: float, d_model:int) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout,d_model) for _ in range(2)])
    
    def forward(self, x, src_mask): 
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self,layers:nn.ModuleList, d_model:int) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self,self_attention_block: MultiHeadAttentionBlock, cross_attention_block:MultiHeadAttentionBlock, feed_forward_block: FeedforwardBlock, dropout:float, d_model:int ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block

        self.residual_connections = nn.ModuleList([ResidualConnection(dropout,d_model) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x:self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connections[1](x, lambda x:self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList,d_model:int)-> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model : int, vocab_size: int) -> None :
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, vocab_size)        
        return torch.log_softmax(self.proj(x), dim=-1)

class TransformerBlock(nn.Module):
    def __init__(self, encoder:Encoder, decoder:Decoder, src_embed:InputEmbedding, tgt_embed:InputEmbedding, src_pos:PositionalEncoding, tgt_pos:PositionalEncoding, projection_layer:ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self,x):
        return self.projection_layer(x)

def build_transformer(src_vocab_size:int, tgt_vocab_size:int, src_seq_len:int, tgt_seq_len:int,d_model:int=512,N:int = 6, h:int = 8, dropout: float = 0.1, d_ff:int =2048):
    # Create the embeding Layers
    src_embed = InputEmbedding(src_vocab_size, d_model)
    tgt_embed = InputEmbedding(tgt_vocab_size, d_model)

    # Create the Positional Encodign Layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block = FeedforwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout, d_model)
        encoder_blocks.append(encoder_block)
    
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedforwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout, d_model)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks),d_model)
    decoder = Decoder(nn.ModuleList(decoder_blocks),d_model)

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model,tgt_vocab_size)

    # Create the transformer
    transformer = TransformerBlock(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer

