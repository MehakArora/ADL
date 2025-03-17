import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import matplotlib.pyplot as plt

class CustomMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, debug = False):
        super(CustomMultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.debug = debug
        self.embed_dim = embed_dim # embed_dim is actually qvk dim x num_heads
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)
        
        if self.debug:
            print(f"Query shape: {query.shape}")
            print(f"Key shape: {key.shape}")
            print(f"Value shape: {value.shape}")
            if attn_mask is not None:
                print(f"Attention mask shape: {attn_mask.shape}")
                print(f"Attention mask: {attn_mask}")
        
        # Linear projections and reshape for multi-head attention
        # [batch_size, seq_len, embed_dim] -> [batch_size, seq_len, num_heads, head_dim]
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim)
        
        if self.debug:
            print(f"Query shape after projection: {q.shape}")
            print(f"Query: {q}")    
            print(f"Key shape after projection: {k.shape}")
            print(f"Key: {k}")
            print(f"Value shape after projection: {v.shape}")
            print(f"Value: {v}")

        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if self.debug:
            print(f"Query shape after transpose: {q.shape}")
            print(f"Key shape after transpose: {k.shape}")
            print(f"Value shape after transpose: {v.shape}")

        # Compute attention scores
        # [batch_size, num_heads, query_len, key_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.debug:
            print(f"Scores shape: {scores.shape}")
            print(f"Scores: {scores}")


        # Apply mask if provided
        if attn_mask is not None:
            # For 4D mask [batch_size, 1, 1, seq_len] -> [batch_size, 1, 1, seq_len]
            # Broadcast to [batch_size, num_heads, query_len, key_len]
            scores = scores.masked_fill_(attn_mask == torch.tensor(False), float("-inf"))
            
        # Apply softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        if self.debug:
            print(f"Attention weights shape: {attn_weights.shape}")
            print(f"Attention weights: {attn_weights}")

        # Apply attention weights to values
        # [batch_size, num_heads, seq_len, head_dim]
        context = torch.matmul(attn_weights, v)
        
        if self.debug:
            print(f"Context shape: {context.shape}")

        # Transpose back and reshape
        # [batch_size, seq_len, embed_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        if self.debug:
            print(f"Context shape before final projection: {context.shape}")
        
        # Final projection
        output = self.out_proj(context)
        if self.debug:
            print(f"Output of MHA shape: {output.shape}")

        return output, attn_weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim, mask_zero=False, debug = False):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim, max_len)
        self.mask_zero = mask_zero
        self.debug = debug 
        
    def forward(self, x):
        # Create mask if required
        mask = None
        if self.mask_zero:
            mask = (x != 0).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            if self.debug:
                print(f"Mask shape: {mask.shape}")
                print(f"Mask: {mask}")
            
        if self.debug:
            print(f"Input shape: {x.shape}")
            
        # Get token embeddings
        x = self.token_embedding(x)
        
        if self.debug:
            print(f"Token embeddings shape: {x.shape}")

        # Add positional encoding
        x = self.position_encoding(x)

        if self.debug:
            print(f"Positional embeddings shape: {x.shape}")
        
        return x, mask

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1, debug = False):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = CustomMultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()

        self.debug = debug
        
    def forward(self, src, src_mask=None):
        # Self attention + residual connection + normalization

        if self.debug:
            print(f"Input shape: {src.shape}")
            if src_mask is not None:
                print(f"Mask shape: {src_mask.shape}")
        
        src = self.norm1(src)
        src2, weights = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        
        
        # Feed forward + residual connection + normalization
        src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        
        
        return src, weights

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers=1, dim_feedforward = 2048, dropout =0.1, debug = False):
        super(TransformerEncoder, self).__init__()
        self.debug = debug
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout = dropout, debug=self.debug) 
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        # PyTorch expects sequence first, batch second for attention layers
        #x = x.transpose(0, 1)
        
        if self.debug:
            print(f"Input shape: {x.shape}")
            if mask is not None:
                print(f"Mask shape: {mask.shape}")
        for layer in self.layers:
            x, weights = layer(x, src_mask=mask)
            if self.debug:
                print(f"Encoder Layer Output shape: {x.shape}")
            
        # Return to batch first, sequence second
        if self.debug:
            print(f"Encoder Output after transpose: {x.shape}")
        return x.transpose(0, 1), weights

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1, debug = False):
        super(TransformerDecoderLayer, self).__init__()
        self.debug = debug
        self.self_attn = CustomMultiHeadAttention(embed_dim, num_heads, dropout=dropout, debug = self.debug)
        self.multihead_attn = CustomMultiHeadAttention(embed_dim, num_heads, dropout=dropout, debug = self.debug)
        
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self attention + residual connection + normalization
        if self.debug:
            print("Target shape: ", tgt.shape)
            print("Memory shape: ", memory.shape)
            if tgt_mask is not None:
                print("Target mask shape: ", tgt_mask.shape)
            
            if memory_mask is not None:
                print("Memory mask shape: ", memory_mask.shape)
        
        batch_size, seq_len, _ = tgt.shape
        
        # Generate causal mask for self-attention
        causal_mask = ~torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(tgt.device)
        
        if self.debug:
            print("Causal mask shape: ", causal_mask.shape)
            print("Causal mask: ", causal_mask)
        
        if tgt_mask is not None:
            tgt_mask = tgt_mask & causal_mask  # Combine padding mask with causal mask
        else:
            tgt_mask = causal_mask
        
        if self.debug:
            print("Mask:")
            print(tgt_mask.shape)
        
        tgt = self.norm1(tgt)
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        
        
        if self.debug:
            print("After Self Attention")
            print(tgt)
        
        # Cross attention + residual connection + normalization
        tgt = self.norm2(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        
        
        if self.debug:
            print("After Cross Attention")
            print(tgt)
            
        # Feed forward + residual connection + normalization
        tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        
        
        if self.debug:
            print("Decoder Output shape: ", tgt.shape)

        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers=1, dim_feedforward = 2048, dropout = 0.1, debug = False):
        super(TransformerDecoder, self).__init__()
        self.debug = debug
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, dim_feedforward=dim_feedforward, dropout = dropout, debug = self.debug)
            for _ in range(num_layers)
        ])
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        
        if self.debug:
            print("Input to the decoder shape: {} and {}".format(tgt.shape, memory.shape))
            if tgt_mask is not None:
                print("Target mask shape: ", tgt_mask.shape)
            if memory_mask is not None:
                print("Memory mask shape: ", memory_mask.shape)

        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
            
        # Return to batch first, sequence second
        if self.debug:
            print("Decoder output shape after transpose: ", tgt.shape)

        return tgt.transpose(0, 1)

class TransformerModel(nn.Module):
    def __init__(self, en_vocab_size, fr_vocab_size, embed_dim, num_heads,  sequence_len,num_layers =4, dim_feedforward=2048, dropout=0.2, debug = False):
        super(TransformerModel, self).__init__()
        self.debug = debug
        # Encoder
        self.encoder_embedding = TokenAndPositionEmbedding(en_vocab_size, sequence_len, embed_dim, debug = self.debug)
        self.encoder = TransformerEncoder(embed_dim, num_heads, num_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout, debug = self.debug)
        
        # Decoder
        self.decoder_embedding = TokenAndPositionEmbedding(fr_vocab_size, sequence_len, embed_dim,  mask_zero=True, debug = self.debug)
        self.decoder = TransformerDecoder(embed_dim, num_heads,num_layers=num_layers, dim_feedforward=dim_feedforward, dropout = dropout, debug = self.debug)
        
        # Output layer
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(embed_dim, fr_vocab_size)
        self.init_params()
        #self.apply(self.init_weights)

    def init_params(self, default_initialization=False):
        if not default_initialization:
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            nn.init.xavier_normal_(self.output_layer.weight, gain=0.01)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.zeros_(m.bias)
            torch.nn.init.ones_(m.weight)
            
    def forward(self, encoder_input, decoder_input):
        # Encoder
        encoder_embeddings, _ = self.encoder_embedding(encoder_input)
        if self.debug:
            print("Encoder Embeddings shape: " , encoder_embeddings.shape)
        encoder_output, weights = self.encoder(encoder_embeddings)
        if self.debug:
            print("Encoder Output shape: " , encoder_output.shape)
            
        # Decoder
        decoder_embeddings, decoder_mask = self.decoder_embedding(decoder_input)
        if self.debug:
            print("Decoder Embeddings shape: " , decoder_embeddings.shape)
            print("Decoder Mask shape: " , decoder_mask.shape)

        decoder_output = self.decoder(decoder_embeddings, encoder_output, tgt_mask=decoder_mask)
        if self.debug:
            print("Decoder Output shape: " , decoder_output.shape)
            
        # Output
        decoder_output = self.dropout(decoder_output)
        return self.output_layer(decoder_output), weights

def analyze_state_dict_shapes_and_names(model):
    # This part helped me figure out that I don't have positional encodings saved in the state dict
    print(model.state_dict().keys())

    # This part helped me see that src MHA was missing in the decoder since both it and trg MHA were referencing
    # the same MHA object in memory - stupid mistake, happens all the time, embrace the suck!
    for name, param in model.named_parameters():
        print(name, param.shape)
        if not param.requires_grad:
            raise Exception('Expected all of the params to be trainable - no param freezing used.')