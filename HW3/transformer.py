import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import copy
from utils import visualize_all_attention_heads

class CustomMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, debug=False, name="MHA"):
        super(CustomMultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.debug = debug
        self.name = name  # Add name for identifying the MHA instance in debug prints
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
        
        # Store intermediate values for debugging
        self.intermediates = {}
        
    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)
        
        if self.debug:
            print(f"[{self.name}] Input shapes - Query: {query.shape}, Key: {key.shape}, Value: {value.shape}")
            if attn_mask is not None:
                print(f"[{self.name}] Attention mask shape: {attn_mask.shape}")
                print(f"[{self.name}] Attention mask sample:\n{attn_mask[0, 0, :5, :5]}")
        
        # Linear projections and reshape for multi-head attention
        # [batch_size, seq_len, embed_dim] -> [batch_size, seq_len, num_heads, head_dim]
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Store projections for debugging
        self.intermediates['q_proj'] = q.detach().clone()
        self.intermediates['k_proj'] = k.detach().clone()
        self.intermediates['v_proj'] = v.detach().clone()
        
        if self.debug:
            print(f"[{self.name}] Projections - Q: {q.shape}, K: {k.shape}, V: {v.shape}")
            print(f"[{self.name}] Q projection sample:\n{q[0, 0, 0, :5]}")
            print(f"[{self.name}] K projection sample:\n{k[0, 0, 0, :5]}")
            print(f"[{self.name}] V projection sample:\n{v[0, 0, 0, :5]}")

        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        self.intermediates['q_transposed'] = q.detach().clone()
        self.intermediates['k_transposed'] = k.detach().clone()
        self.intermediates['v_transposed'] = v.detach().clone()
        
        if self.debug:
            print(f"[{self.name}] After transpose - Q: {q.shape}, K: {k.shape}, V: {v.shape}")

        # Compute attention scores
        # [batch_size, num_heads, query_len, key_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        self.intermediates['raw_scores'] = scores.detach().clone()
        
        if self.debug:
            print(f"[{self.name}] Attention scores shape: {scores.shape}")
            print(f"[{self.name}] Attention scores sample:\n{scores[0, 0, :5, :5]}")
            print(f"[{self.name}] Attention scores stats - Min: {scores.min().item()}, Max: {scores.max().item()}, Mean: {scores.mean().item()}")

        # Apply mask if provided
        if attn_mask is not None:
            # For 4D mask [batch_size, 1, 1, seq_len] -> [batch_size, 1, 1, seq_len]
            # Broadcast to [batch_size, num_heads, query_len, key_len]
            scores = scores.masked_fill_(attn_mask == torch.tensor(False), float("-inf"))
            self.intermediates['masked_scores'] = scores.detach().clone()
            
            if self.debug:
                print(f"[{self.name}] Masked scores sample:\n{scores[0, 0, :5, :5]}")
        
        # Apply softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)
        self.intermediates['attn_weights_pre_dropout'] = attn_weights.detach().clone()
        
        attn_weights = self.dropout(attn_weights)
        self.intermediates['attn_weights'] = attn_weights.detach().clone()
        
        if self.debug:
            print(f"[{self.name}] Attention weights shape: {attn_weights.shape}")
            print(f"[{self.name}] Attention weights sample:\n{attn_weights[0, 0, :5, :5]}")
            print(f"[{self.name}] Attention weight stats - Min: {attn_weights.min().item()}, Max: {attn_weights.max().item()}, Mean: {attn_weights.mean().item()}")

        # Apply attention weights to values
        # [batch_size, num_heads, seq_len, head_dim]
        context = torch.matmul(attn_weights, v)
        self.intermediates['context'] = context.detach().clone()
        
        if self.debug:
            print(f"[{self.name}] Context shape: {context.shape}")
            print(f"[{self.name}] Context sample:\n{context[0, 0, 0, :5]}")

        # Transpose back and reshape
        # [batch_size, seq_len, embed_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        self.intermediates['context_reshaped'] = context.detach().clone()
        
        if self.debug:
            print(f"[{self.name}] Context after reshape: {context.shape}")
        
        # Final projection
        output = self.out_proj(context)
        self.intermediates['output'] = output.detach().clone()
        
        if self.debug:
            print(f"[{self.name}] Final output shape: {output.shape}")
            print(f"[{self.name}] Output sample:\n{output[0, 0, :5]}")

        return output, attn_weights

    def register_hooks(self):
        """Register hooks to capture gradients flowing through the MHA"""
        self.gradients = {}
        
        def save_grad(name):
            def hook(grad):
                self.gradients[name] = grad.detach().clone()
            return hook
        
        self.q_proj.weight.register_hook(save_grad('q_proj_weight'))
        self.k_proj.weight.register_hook(save_grad('k_proj_weight'))
        self.v_proj.weight.register_hook(save_grad('v_proj_weight'))
        self.out_proj.weight.register_hook(save_grad('out_proj_weight'))
        
        return self

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, debug=False):
        super(PositionalEncoding, self).__init__()
        self.debug = debug
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
        # Intermediate values
        self.intermediates = {}

    def forward(self, x):
        if self.debug:
            print(f"[PositionalEncoding] Input shape: {x.shape}")
            print(f"[PositionalEncoding] PE slice used shape: {self.pe[:, :x.size(1), :].shape}")
        
        # Store raw input and positional encoding used
        self.intermediates['input'] = x.detach().clone()
        self.intermediates['pe_slice'] = self.pe[:, :x.size(1), :].detach().clone()
        
        output = x + self.pe[:, :x.size(1), :]
        self.intermediates['output'] = output.detach().clone()
        
        if self.debug:
            print(f"[PositionalEncoding] Output shape: {output.shape}")
            print(f"[PositionalEncoding] Input sample: {x[0, 0, :5]}")
            print(f"[PositionalEncoding] PE sample: {self.pe[0, 0, :5]}")
            print(f"[PositionalEncoding] Output sample: {output[0, 0, :5]}")
        
        return output

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim, mask_zero=False, debug=False, name="Embedding"):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim, max_len, debug=debug)
        self.mask_zero = mask_zero
        self.debug = debug
        self.name = name
        
        # Intermediate values
        self.intermediates = {}
        
    def forward(self, x):
        # Create mask if required
        mask = None
        if self.mask_zero:
            mask = (x != 0).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            self.intermediates['mask'] = mask.detach().clone()
            
            if self.debug:
                print(f"[{self.name}] Mask shape: {mask.shape}")
                print(f"[{self.name}] Mask sample:\n{mask[0, 0, 0, :10]}")
            
        if self.debug:
            print(f"[{self.name}] Input shape: {x.shape}")
            print(f"[{self.name}] Input sample (token ids):\n{x[0, :10]}")
            
        self.intermediates['input'] = x.detach().clone()
        
        # Get token embeddings
        x = self.token_embedding(x)
        self.intermediates['token_embedding'] = x.detach().clone()
        
        if self.debug:
            print(f"[{self.name}] Token embeddings shape: {x.shape}")
            print(f"[{self.name}] Token embeddings sample:\n{x[0, 0, :5]}")

        # Add positional encoding
        x = self.position_encoding(x)
        self.intermediates['position_encoding'] = x.detach().clone()

        if self.debug:
            print(f"[{self.name}] Final embeddings shape: {x.shape}")
            print(f"[{self.name}] Final embeddings sample:\n{x[0, 0, :5]}")
        
        return x, mask

    def register_hooks(self):
        """Register hooks to capture gradients flowing through the embedding layer"""
        self.gradients = {}
        
        def save_grad(name):
            def hook(grad):
                self.gradients[name] = grad.detach().clone()
            return hook
        
        self.token_embedding.weight.register_hook(save_grad('token_embedding_weight'))
        
        return self

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1, debug=False, layer_idx=0):
        super(TransformerEncoderLayer, self).__init__()
        self.layer_idx = layer_idx
        self.debug = debug
        self.name = f"EncoderLayer_{layer_idx}"
        
        self.self_attn = CustomMultiHeadAttention(embed_dim, num_heads, dropout=dropout, debug=debug, name=f"{self.name}_SelfAttn")
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()
        
        # To store intermediate values
        self.intermediates = {}

    def forward(self, src, src_mask=None):
        if self.debug:
            print(f"\n[{self.name}] Input shape: {src.shape}")
            if src_mask is not None:
                print(f"[{self.name}] Mask shape: {src_mask.shape}")
        
        self.intermediates['input'] = src.detach().clone()
        
        # First normalization
        src_norm1 = self.norm1(src)
        self.intermediates['norm1_out'] = src_norm1.detach().clone()
        
        if self.debug:
            print(f"[{self.name}] After first norm: {src_norm1.shape}")
            print(f"[{self.name}] Norm1 stats - Min: {src_norm1.min().item()}, Max: {src_norm1.max().item()}, Mean: {src_norm1.mean().item()}")
        
        # Self attention
        src2, weights = self.self_attn(src_norm1, src_norm1, src_norm1, attn_mask=src_mask)
        self.intermediates['self_attn_out'] = src2.detach().clone()
        self.intermediates['attn_weights'] = weights.detach().clone()
        
        if self.debug:
            print(f"[{self.name}] After self attention: {src2.shape}")
        
        # First residual connection
        src_res1 = src + self.dropout1(src2)
        self.intermediates['residual1_out'] = src_res1.detach().clone()
        
        if self.debug:
            print(f"[{self.name}] After first residual: {src_res1.shape}")
        
        # Second normalization
        src_norm2 = self.norm2(src_res1)
        self.intermediates['norm2_out'] = src_norm2.detach().clone()
        
        if self.debug:
            print(f"[{self.name}] After second norm: {src_norm2.shape}")
        
        # Feed forward network
        ff_out = self.linear1(src_norm2)
        self.intermediates['ff1_out'] = ff_out.detach().clone()
        
        ff_out = self.activation(ff_out)
        self.intermediates['ff_activation_out'] = ff_out.detach().clone()
        
        ff_out = self.dropout(ff_out)
        ff_out = self.linear2(ff_out)
        self.intermediates['ff2_out'] = ff_out.detach().clone()
        
        if self.debug:
            print(f"[{self.name}] After feed forward: {ff_out.shape}")
            print(f"[{self.name}] FFN stats - Min: {ff_out.min().item()}, Max: {ff_out.max().item()}, Mean: {ff_out.mean().item()}")
        
        # Second residual connection
        src_out = src_res1 + self.dropout2(ff_out)
        self.intermediates['output'] = src_out.detach().clone()
        
        if self.debug:
            print(f"[{self.name}] Output shape: {src_out.shape}")
            print(f"[{self.name}] Output stats - Min: {src_out.min().item()}, Max: {src_out.max().item()}, Mean: {src_out.mean().item()}")
        
        return src_out, weights

    def register_hooks(self):
        """Register hooks to capture gradients flowing through the encoder layer"""
        self.gradients = {}
        
        def save_grad(name):
            def hook(grad):
                self.gradients[name] = grad.detach().clone()
            return hook
        
        self.linear1.weight.register_hook(save_grad('linear1_weight'))
        self.linear2.weight.register_hook(save_grad('linear2_weight'))
        self.norm1.weight.register_hook(save_grad('norm1_weight'))
        self.norm2.weight.register_hook(save_grad('norm2_weight'))
        
        # Also register hooks for the attention mechanism
        self.self_attn.register_hooks()
        
        return self

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers=1, dim_feedforward=2048, dropout=0.1, debug=False):
        super(TransformerEncoder, self).__init__()
        self.debug = debug
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim, 
                num_heads, 
                dim_feedforward, 
                dropout=dropout, 
                debug=debug,
                layer_idx=i
            ) 
            for i in range(num_layers)
        ])
        
        # Store attention weights from all layers
        self.all_attentions = []
        
        # Intermediates
        self.intermediates = {}
        self.gradients = {}
        
    def forward(self, x, mask=None):
        if self.debug:
            print("\n[Encoder] Input shape:", x.shape)
            if mask is not None:
                print("[Encoder] Mask shape:", mask.shape)
        
        self.intermediates['input'] = x.detach().clone()
        
        # Reset attention weights
        self.all_attentions = []
        
        # Store layer outputs and intermediate states
        layer_outputs = []
        layer_intermediates = []
        
        # Process through each layer
        for i, layer in enumerate(self.layers):
            x, weights = layer(x, src_mask=mask)
            self.all_attentions.append(weights)
            
            # Store per-layer output and intermediates
            layer_outputs.append(x.detach().clone())
            layer_intermediates.append(copy.deepcopy(layer.intermediates))
            
            if self.debug:
                print(f"[Encoder] Layer {i} output shape: {x.shape}")
                print(f"[Encoder] Layer {i} attention shape: {weights.shape}")
        
        self.intermediates['output'] = x.detach().clone()
        self.intermediates['layer_outputs'] = layer_outputs
        self.intermediates['layer_intermediates'] = layer_intermediates
        
        if self.debug:
            print("[Encoder] Final output shape:", x.shape)
            
        return x, self.all_attentions
        
    def register_hooks(self):
        """Register hooks for all encoder layers"""
        self.gradients = {}
        
        def save_grad(name):
            def hook(grad):
                self.gradients[name] = grad.detach().clone()
            return hook
            
        # Register hooks for each layer
        for i, layer in enumerate(self.layers):
            layer.register_hooks()
            
        return self
    
    def get_attention_maps(self):
        """Return attention maps from all layers"""
        return self.all_attentions
    
    def get_intermediates(self):
        """Return all intermediate tensors from encoder and its layers"""
        return {
            'encoder': self.intermediates,
            'layers': [layer.intermediates for layer in self.layers]
        }
    
    def get_gradients(self):
        """Return all gradients from encoder and its layers"""
        return {
            'encoder': self.gradients,
            'layers': [layer.gradients for layer in self.layers if hasattr(layer, 'gradients')]
        }

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1, debug=False, layer_idx=0):
        super(TransformerDecoderLayer, self).__init__()
        self.debug = debug
        self.layer_idx = layer_idx
        self.name = f"DecoderLayer_{layer_idx}"
        
        # Create separate attention instances for self and cross attention
        self.self_attn = CustomMultiHeadAttention(
            embed_dim, num_heads, dropout=dropout, debug=debug, name=f"{self.name}_SelfAttn"
        )
        self.cross_attn = CustomMultiHeadAttention(
            embed_dim, num_heads, dropout=dropout, debug=debug, name=f"{self.name}_CrossAttn"
        )
        
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
        
        # Intermediates
        self.intermediates = {}
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        if self.debug:
            print(f"\n[{self.name}] Target shape: {tgt.shape}")
            print(f"[{self.name}] Memory shape: {memory.shape}")
            if tgt_mask is not None:
                print(f"[{self.name}] Target mask shape: {tgt_mask.shape}")
            if memory_mask is not None:
                print(f"[{self.name}] Memory mask shape: {memory_mask.shape}")
        
        self.intermediates['input_tgt'] = tgt.detach().clone()
        self.intermediates['input_memory'] = memory.detach().clone()
        
        batch_size, seq_len, _ = tgt.shape
        
        # Generate causal mask for self-attention if not provided
        causal_mask = torch.triu(torch.ones(seq_len, seq_len) == 1).transpose(0, 1)
        causal_mask = causal_mask.float().masked_fill(causal_mask == 0, float('-inf')).masked_fill(causal_mask == 1, float(0.0))
        causal_mask = causal_mask.to(tgt.device)
        
        self.intermediates['causal_mask'] = causal_mask.detach().clone()
        
        if self.debug:
            print(f"[{self.name}] Causal mask shape: {causal_mask.shape}")
            print(f"[{self.name}] Causal mask sample:\n{causal_mask[:5, :5]}")
        
        # First normalization
        tgt_norm1 = self.norm1(tgt)
        self.intermediates['norm1_out'] = tgt_norm1.detach().clone()
        
        # Self attention with causal mask
        # Apply causal mask to prevent attending to future tokens
        if tgt_mask is not None:
            # Combine padding mask with causal mask
            combined_mask = causal_mask.unsqueeze(0).unsqueeze(1) + tgt_mask
            tgt2, self_attn_weights = self.self_attn(tgt_norm1, tgt_norm1, tgt_norm1, attn_mask=combined_mask)
        else:
            tgt2, self_attn_weights = self.self_attn(tgt_norm1, tgt_norm1, tgt_norm1, attn_mask=causal_mask.unsqueeze(0).unsqueeze(1))
        
        self.intermediates['self_attn_out'] = tgt2.detach().clone()
        self.intermediates['self_attn_weights'] = self_attn_weights.detach().clone()
        
        if self.debug:
            print(f"[{self.name}] After self attention: {tgt2.shape}")
        
        # First residual connection
        tgt = tgt + self.dropout1(tgt2)
        self.intermediates['residual1_out'] = tgt.detach().clone()
        
        # Second normalization
        tgt_norm2 = self.norm2(tgt)
        self.intermediates['norm2_out'] = tgt_norm2.detach().clone()
        
        # Cross attention
        tgt2, cross_attn_weights = self.cross_attn(tgt_norm2, memory, memory, attn_mask=memory_mask)
        self.intermediates['cross_attn_out'] = tgt2.detach().clone()
        self.intermediates['cross_attn_weights'] = cross_attn_weights.detach().clone()
        
        if self.debug:
            print(f"[{self.name}] After cross attention: {tgt2.shape}")
        
        # Second residual connection
        tgt = tgt + self.dropout2(tgt2)
        self.intermediates['residual2_out'] = tgt.detach().clone()
        
        # Third normalization
        tgt_norm3 = self.norm3(tgt)
        self.intermediates['norm3_out'] = tgt_norm3.detach().clone()
        
        # Feed forward network
        ff_out = self.linear1(tgt_norm3)
        self.intermediates['ff1_out'] = ff_out.detach().clone()
        
        ff_out = self.activation(ff_out)
        self.intermediates['ff_activation_out'] = ff_out.detach().clone()
        
        ff_out = self.dropout(ff_out)
        ff_out = self.linear2(ff_out)
        self.intermediates['ff2_out'] = ff_out.detach().clone()
        
        if self.debug:
            print(f"[{self.name}] After feed forward: {ff_out.shape}")
        
        # Third residual connection
        tgt = tgt + self.dropout3(ff_out)
        self.intermediates['output'] = tgt.detach().clone()
        
        if self.debug:
            print(f"[{self.name}] Final output shape: {tgt.shape}")
        
        return tgt, self_attn_weights, cross_attn_weights

    def register_hooks(self):
        """Register hooks to capture gradients flowing through the decoder layer"""
        self.gradients = {}
        
        def save_grad(name):
            def hook(grad):
                self.gradients[name] = grad.detach().clone()
            return hook
        
        self.linear1.weight.register_hook(save_grad('linear1_weight'))
        self.linear2.weight.register_hook(save_grad('linear2_weight'))
        self.norm1.weight.register_hook(save_grad('norm1_weight'))
        self.norm2.weight.register_hook(save_grad('norm2_weight'))
        self.norm3.weight.register_hook(save_grad('norm3_weight'))
        
        # Also register hooks for both attention mechanisms
        self.self_attn.register_hooks()
        self.cross_attn.register_hooks()
        
        return self

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers=1, dim_feedforward=2048, dropout=0.1, debug=False):
        super(TransformerDecoder, self).__init__()
        self.debug = debug
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                embed_dim, 
                num_heads, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout, 
                debug=debug,
                layer_idx=i
            )
            for i in range(num_layers)
        ])
        
        # Store attention weights from all layers
        self.self_attentions = []
        self.cross_attentions = []
        
        # Intermediates and gradients
        self.intermediates = {}
        self.gradients = {}
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        if self.debug:
            print("\n[Decoder] Input shapes - Target:", tgt.shape, "Memory:", memory.shape)
            if tgt_mask is not None:
                print("[Decoder] Target mask shape:", tgt_mask.shape)
            if memory_mask is not None:
                print("[Decoder] Memory mask shape:", memory_mask.shape)
        
        self.intermediates['input_tgt'] = tgt.detach().clone()
        self.intermediates['input_memory'] = memory.detach().clone()
        
        # Reset attention weights
        self.self_attentions = []
        self.cross_attentions = []
        
        # Store per-layer data
        layer_outputs = []
        layer_intermediates = []

        # Process through each layer
        for i, layer in enumerate(self.layers):
            tgt, self_attn, cross_attn = layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
            self.self_attentions.append(self_attn)
            self.cross_attentions.append(cross_attn)
            
            # Store per-layer output and intermediates
            layer_outputs.append(tgt.detach().clone())
            layer_intermediates.append(copy.deepcopy(layer.intermediates))
            
            if self.debug:
                print(f"[Decoder] Layer {i} output shape: {tgt.shape}")
                print(f"[Decoder] Layer {i} self-attention shape: {self_attn.shape}")
                print(f"[Decoder] Layer {i} cross-attention shape: {cross_attn.shape}")
                
        self.intermediates['output'] = tgt.detach().clone()
        self.intermediates['layer_outputs'] = layer_outputs
        self.intermediates['layer_intermediates'] = layer_intermediates
        
        if self.debug:
            print("[Decoder] Final output shape:", tgt.shape)
            
        return tgt, self.self_attentions, self.cross_attentions
        
    def register_hooks(self):
        """Register hooks for all decoder layers"""
        self.gradients = {}
        
        def save_grad(name):
            def hook(grad):
                self.gradients[name] = grad.detach().clone()
            return hook
            
        # Register hooks for each layer
        for i, layer in enumerate(self.layers):
            layer.register_hooks()
            
        return self
    
    def get_attention_maps(self):
        """Return attention maps from all layers"""
        return {
            'self_attention': self.self_attentions,
            'cross_attention': self.cross_attentions
        }
    
    def get_intermediates(self):
        """Return all intermediate tensors from decoder and its layers"""
        return {
            'decoder': self.intermediates,
            'layers': [layer.intermediates for layer in self.layers]
        }
    
    def get_gradients(self):
        """Return all gradients from decoder and its layers"""
        return {
            'decoder': self.gradients,
            'layers': [layer.gradients for layer in self.layers if hasattr(layer, 'gradients')]
        }

class TransformerModel(nn.Module):
    def __init__(self, en_vocab_size, fr_vocab_size, embed_dim, num_heads, sequence_len, 
                 num_layers=4, dim_feedforward=2048, dropout=0.2, debug=False):
        super(TransformerModel, self).__init__()
        self.debug = debug
        
        # Encoder
        self.encoder_embedding = TokenAndPositionEmbedding(
            en_vocab_size, sequence_len, embed_dim, debug=debug, name="EncoderEmbedding"
        )
        self.encoder = TransformerEncoder(
            embed_dim, num_heads, num_layers=num_layers, dim_feedforward=dim_feedforward, 
            dropout=dropout, debug=debug
        )
        
        # Decoder
        self.decoder_embedding = TokenAndPositionEmbedding(
            fr_vocab_size, sequence_len, embed_dim, mask_zero=True, 
            debug=debug, name="DecoderEmbedding"
        )
        self.decoder = TransformerDecoder(
            embed_dim, num_heads, num_layers=num_layers, dim_feedforward=dim_feedforward, 
            dropout=dropout, debug=debug
        )
        
        # Output layer
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(embed_dim, fr_vocab_size)
        
        # Initialize parameters
        self.init_weights()
        
        # Store intermediate values and gradients
        self.intermediates = {}
        self.gradients = {}
        
    def init_params(self, default_initialization=False):
        if not default_initialization:
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            nn.init.xavier_normal_(self.output_layer.weight, gain=0.01)
    
    def init_weights(self):
        for m in self.modules():
            #print(m)
            self.init_module_weights(m)

    def init_module_weights(self, m):        
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.zeros_(m.bias)
            torch.nn.init.ones_(m.weight)
    
    def register_hooks(self):
        """Register hooks for all model components to track gradients"""
        self.gradients = {}
        
        # Register for output layer
        def save_grad(name):
            def hook(grad):
                self.gradients[name] = grad.detach().clone()
            return hook
        
        self.output_layer.weight.register_hook(save_grad('output_layer_weight'))
        
        # Register for encoder and decoder components
        self.encoder_embedding.register_hooks()
        for layer in self.encoder.layers:
            layer.register_hooks()
            
        self.decoder_embedding.register_hooks()
        for layer in self.decoder.layers:
            layer.register_hooks()
            
        return self
            
    def forward(self, encoder_input, decoder_input):
        # Store inputs
        self.intermediates['encoder_input'] = encoder_input.detach().clone()
        self.intermediates['decoder_input'] = decoder_input.detach().clone()
        
        if self.debug:
            print("\n[TransformerModel] Input shapes - Encoder:", encoder_input.shape, "Decoder:", decoder_input.shape)
        
        # Encoder
        encoder_embeddings, _ = self.encoder_embedding(encoder_input)
        self.intermediates['encoder_embeddings'] = encoder_embeddings.detach().clone()
        
        if self.debug:
            print("[TransformerModel] Encoder Embeddings shape:", encoder_embeddings.shape)
            
        encoder_output, encoder_attentions = self.encoder(encoder_embeddings)
        self.intermediates['encoder_output'] = encoder_output.detach().clone()
        self.intermediates['encoder_attentions'] = [attn.detach().clone() for attn in encoder_attentions]
        
        if self.debug:
            print("[TransformerModel] Encoder Output shape:", encoder_output.shape)
            print(f"[TransformerModel] Captured {len(encoder_attentions)} encoder attention maps")
            
        # Decoder
        decoder_embeddings, decoder_mask = self.decoder_embedding(decoder_input)
        self.intermediates['decoder_embeddings'] = decoder_embeddings.detach().clone()
        
        if self.debug:
            print("[TransformerModel] Decoder Embeddings shape:", decoder_embeddings.shape)
            if decoder_mask is not None:
                print("[TransformerModel] Decoder Mask shape:", decoder_mask.shape)

        decoder_output, decoder_self_attentions, decoder_cross_attentions = self.decoder(
            decoder_embeddings, encoder_output, tgt_mask=decoder_mask
        )
        self.intermediates['decoder_output'] = decoder_output.detach().clone()
        self.intermediates['decoder_self_attentions'] = [attn.detach().clone() for attn in decoder_self_attentions]
        self.intermediates['decoder_cross_attentions'] = [attn.detach().clone() for attn in decoder_cross_attentions]
        
        if self.debug:
            print("[TransformerModel] Decoder Output shape:", decoder_output.shape)
            print(f"[TransformerModel] Captured {len(decoder_self_attentions)} decoder self-attention maps")
            print(f"[TransformerModel] Captured {len(decoder_cross_attentions)} decoder cross-attention maps")
            
        # Output projection
        decoder_output = self.dropout(decoder_output)
        logits = self.output_layer(decoder_output)
        self.intermediates['logits'] = logits.detach().clone()
        
        if self.debug:
            print("[TransformerModel] Final logits shape:", logits.shape)
        
        return logits, {
            'encoder_attentions': encoder_attentions,
            'decoder_self_attentions': decoder_self_attentions,
            'decoder_cross_attentions': decoder_cross_attentions
        }

    def get_attention_maps(self):
        """Return a dictionary of all attention maps in the model"""
        if not hasattr(self, 'intermediates') or 'encoder_attentions' not in self.intermediates:
            return None
            
        # Get attention maps from the entire model
        model_attention_maps = {
            'encoder_attentions': self.intermediates['encoder_attentions'],
            'decoder_self_attentions': self.intermediates['decoder_self_attentions'],
            'decoder_cross_attentions': self.intermediates['decoder_cross_attentions']
        }
        
        return {
            'model': model_attention_maps
        }
        
    def get_intermediates(self):
        """Return all intermediate tensors captured during forward pass"""
        # Get intermediates from the main model
        model_intermediates = self.intermediates
        
        # Also get intermediates from submodules
        encoder_intermediates = self.encoder.get_intermediates() if hasattr(self.encoder, 'get_intermediates') else None
        decoder_intermediates = self.decoder.get_intermediates() if hasattr(self.decoder, 'get_intermediates') else None
        encoder_embedding_intermediates = self.encoder_embedding.intermediates if hasattr(self.encoder_embedding, 'intermediates') else None
        decoder_embedding_intermediates = self.decoder_embedding.intermediates if hasattr(self.decoder_embedding, 'intermediates') else None
        
        return {
            'model': model_intermediates,
            'encoder': encoder_intermediates,
            'decoder': decoder_intermediates,
            'encoder_embedding': encoder_embedding_intermediates,
            'decoder_embedding': decoder_embedding_intermediates
        }
    
    def get_gradients(self):
        """Return all gradients captured during backward pass"""
        # Get gradients from the main model
        model_gradients = self.gradients
        
        # Also get gradients from submodules
        encoder_gradients = self.encoder.get_gradients() if hasattr(self.encoder, 'get_gradients') else None
        decoder_gradients = self.decoder.get_gradients() if hasattr(self.decoder, 'get_gradients') else None
        
        # Get gradients for output layer
        output_layer_gradients = {}
        if hasattr(self.output_layer, 'weight') and self.output_layer.weight.grad is not None:
            output_layer_gradients['weight'] = self.output_layer.weight.grad.detach().clone()
        if hasattr(self.output_layer, 'bias') and self.output_layer.bias is not None and self.output_layer.bias.grad is not None:
            output_layer_gradients['bias'] = self.output_layer.bias.grad.detach().clone()
        
        return {
            'model': model_gradients,
            'encoder': encoder_gradients,
            'decoder': decoder_gradients,
            'output_layer': output_layer_gradients
        }


def analyze_state_dict_shapes_and_names(model):
    """Analyze model's state dictionary"""
    print("\n===== MODEL STATE DICT ANALYSIS =====")
    
    # Get state dict
    state_dict = model.state_dict()
    
    # Print keys and shapes
    for name, param in state_dict.items():
        print(f"{name}: {param.shape}")
    
    # Check trainable vs. non-trainable parameters
    trainable_params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    non_trainable_params = {name: param for name, param in model.named_parameters() if not param.requires_grad}
    
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params.values())}")
    print(f"Non-trainable parameters: {sum(p.numel() for p in non_trainable_params.values())}")
    
    # Check for any parameters that aren't trainable
    if non_trainable_params:
        print("\nNon-trainable parameter names:")
        for name in non_trainable_params.keys():
            print(f"- {name}")
    else:
        print("\nAll parameters are trainable")


def visualize_attention(attention_weights, title='Attention Weights', layer_idx=0, head_idx=0, sample_idx=0):
    """
    Visualize attention weights using heatmaps
    
    Args:
        attention_weights: Tensor of shape [batch_size, num_heads, seq_len_q, seq_len_k]
        title: Title for the plot
        layer_idx: Which layer to visualize (if attention_weights is a list)
        head_idx: Which attention head to visualize
        sample_idx: Which batch sample to visualize
    """
    plt.figure(figsize=(10, 8))
    
    # Handle if input is a list of attention matrices (from different layers)
    if isinstance(attention_weights, list):
        attn = attention_weights[layer_idx]
    else:
        attn = attention_weights
    
    # Extract weights for the specified head and sample
    weights = attn[sample_idx, head_idx].detach().cpu().numpy()
    
    # Create heatmap
    ax = sns.heatmap(weights, cmap='viridis', vmin=0, vmax=1)
    
    plt.title(f"{title} (Layer {layer_idx+1}, Head {head_idx+1})")
    plt.xlabel('Key position')
    plt.ylabel('Query position')
    
    return plt.gcf()





def compare_layer_attentions(model_output, layer_type='encoder', num_layers=None, head_idx=0, sample_idx=0):
    """
    Compare attention patterns across different layers
    
    Args:
        model_output: Dictionary containing attention weights from a model forward pass
        layer_type: 'encoder', 'decoder_self', or 'decoder_cross'
        num_layers: Number of layers to visualize (defaults to all)
        head_idx: Which attention head to visualize
        sample_idx: Which batch sample to visualize
    """
    if layer_type == 'encoder':
        attention_key = 'encoder_attentions'
    elif layer_type == 'decoder_self':
        attention_key = 'decoder_self_attentions'
    elif layer_type == 'decoder_cross':
        attention_key = 'decoder_cross_attentions'
    else:
        raise ValueError("layer_type must be 'encoder', 'decoder_self', or 'decoder_cross'")
    
    # Get attention weights
    attention_weights = model_output[attention_key]
    
    # Determine number of layers to visualize
    if num_layers is None or num_layers > len(attention_weights):
        num_layers = len(attention_weights)
    
    # Create figure
    fig, axes = plt.subplots(1, num_layers, figsize=(num_layers * 4, 4))
    if num_layers == 1:
        axes = [axes]
    
    # Plot heatmaps
    for i in range(num_layers):
        weights = attention_weights[i][sample_idx, head_idx].detach().cpu().numpy()
        
        sns.heatmap(weights, cmap='viridis', vmin=0, vmax=1, ax=axes[i], cbar=(i == num_layers-1))
        axes[i].set_title(f"Layer {i+1}")
        
        if i > 0:  # Remove y-axis labels for all but the first plot
            axes[i].set_ylabel('')
            axes[i].set_yticklabels([])
    
    # Add title
    layer_type_name = {
        'encoder': 'Encoder Self-Attention', 
        'decoder_self': 'Decoder Self-Attention', 
        'decoder_cross': 'Decoder Cross-Attention'
    }[layer_type]
    
    plt.suptitle(f"{layer_type_name} Patterns Across Layers (Head {head_idx+1})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig


class GradientTracker:
    """
    A class to track and analyze gradients during training.
    
    Usage:
        tracker = GradientTracker(model)
        
        # During training loop:
        tracker.reset()  # Reset before each backward pass
        loss.backward()  # Do normal backward pass
        tracker.capture()  # Capture gradients after backward
        
        # Analyze gradients:
        tracker.plot_gradient_norms()
        tracker.plot_gradient_histograms()
    """
    def __init__(self, model):
        self.model = model
        self.gradients = {}
        self.gradient_history = defaultdict(list)
        self.iteration = 0
        
        # Register hooks for all parameters
        self.hooks = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(self._make_hook(name))
                self.hooks.append(hook)
    
    def _make_hook(self, name):
        def hook(grad):
            self.gradients[name] = grad.detach().clone()
        return hook
    
    def reset(self):
        """Reset gradients before backward pass"""
        self.gradients = {}
    
    def capture(self):
        """Capture gradients after backward pass"""
        for name, grad in self.gradients.items():
            # Calculate norm
            norm = grad.norm().item()
            self.gradient_history[name].append(norm)
        
        self.iteration += 1
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def plot_gradient_norms(self, top_k=10, last_n=None):
        """
        Plot gradient norms over iterations
        
        Args:
            top_k: Number of parameters with largest gradient norms to plot
            last_n: Number of most recent iterations to plot (None for all)
        """
        # Calculate average norm for each parameter
        avg_norms = {}
        for name, history in self.gradient_history.items():
            if last_n is not None and len(history) > last_n:
                history = history[-last_n:]
            avg_norms[name] = np.mean(history)
        
        # Sort by average norm
        sorted_names = sorted(avg_norms.keys(), key=lambda x: avg_norms[x], reverse=True)
        
        # Keep only top_k
        if top_k is not None:
            sorted_names = sorted_names[:top_k]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        iterations = range(1, self.iteration + 1)
        if last_n is not None:
            iterations = iterations[-last_n:]
        
        for name in sorted_names:
            history = self.gradient_history[name]
            if last_n is not None and len(history) > last_n:
                history = history[-last_n:]
            plt.plot(iterations, history, label=name)
        
        plt.title('Gradient Norms Over Training')
        plt.xlabel('Iteration')
        plt.ylabel('Gradient Norm')
        plt.yscale('log')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_gradient_histograms(self, layer_names=None, bins=50):
        """
        Plot histograms of gradients for specific layers
        
        Args:
            layer_names: List of layer names to plot (None for all)
            bins: Number of histogram bins
        """
        if not self.gradients:
            print("No gradients captured yet.")
            return None
        
        # If no specific names provided, use all
        if layer_names is None:
            layer_names = list(self.gradients.keys())
        
        # Filter out names that don't exist
        layer_names = [name for name in layer_names if name in self.gradients]
        
        if not layer_names:
            print("No valid layer names provided.")
            return None
        
        # Create subplot grid
        n = len(layer_names)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        if rows * cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, name in enumerate(layer_names):
            if i < len(axes):
                grad = self.gradients[name].cpu().flatten().numpy()
                
                # Plot histogram
                axes[i].hist(grad, bins=bins, alpha=0.7)
                axes[i].set_title(name)
                axes[i].set_xlabel('Gradient Value')
                axes[i].set_ylabel('Count')
                
                # Add statistics
                mean = np.mean(grad)
                std = np.std(grad)
                median = np.median(grad)
                max_val = np.max(np.abs(grad))
                
                stats_text = f"Mean: {mean:.2e}\nStd: {std:.2e}\nMedian: {median:.2e}\nMax Abs: {max_val:.2e}"
                axes[i].text(0.95, 0.95, stats_text, transform=axes[i].transAxes, 
                             verticalalignment='top', horizontalalignment='right',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Hide unused subplots
        for i in range(len(layer_names), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Gradient Distributions', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        return fig


# Example usage of the enhanced Transformer model with debugging

def inspect_model_state(model, detailed=False):
    """
    Inspect the model's internal state including all intermediate tensors and attention maps
    
    Args:
        model: The transformer model
        detailed: Whether to print detailed information about tensors
    
    Returns:
        Dictionary containing the model state
    """
    print("\n===== MODEL STATE INSPECTION =====")
    
    # Get all intermediate tensors
    intermediates = model.get_intermediates()
    
    # Get all attention maps
    attention_maps = model.get_attention_maps()
    
    # Get all gradients (if available)
    gradients = model.get_gradients() if hasattr(model, 'get_gradients') else None
    
    # Print summary information
    if intermediates:
        print("\nIntermediate tensors:")
        for module, tensors in intermediates.items():
            if tensors:
                if isinstance(tensors, dict):
                    print(f"  {module}: {len(tensors)} tensors")
                    if detailed:
                        for name, tensor in tensors.items():
                            if isinstance(tensor, torch.Tensor):
                                print(f"    {name}: Shape {tensor.shape}, Min {tensor.min().item():.4f}, Max {tensor.max().item():.4f}, Mean {tensor.mean().item():.4f}")
                            elif isinstance(tensor, list):
                                print(f"    {name}: List of {len(tensor)} items")
                else:
                    print(f"  {module}: Data available but not in dictionary format")
    
    if attention_maps:
        print("\nAttention maps:")
        for module, maps in attention_maps.items():
            if maps:
                if isinstance(maps, dict):
                    for attn_type, attn_list in maps.items():
                        if isinstance(attn_list, list):
                            print(f"  {module} - {attn_type}: {len(attn_list)} layers")
                elif isinstance(maps, list):
                    print(f"  {module}: {len(maps)} layers")
    
    if gradients:
        print("\nGradients:")
        for module, grads in gradients.items():
            if grads:
                if isinstance(grads, dict):
                    print(f"  {module}: {len(grads)} parameters with gradients")
                else:
                    print(f"  {module}: Gradients available but not in dictionary format")
    
    return {
        'intermediates': intermediates,
        'attention_maps': attention_maps,
        'gradients': gradients
    }

def example_forward_pass(model, src, tgt, debug=True):
    """Example of running a forward pass with full debugging and visualization"""
    # Enable debugging if requested
    model.debug = debug
    
    # Register gradient hooks
    model.register_hooks()
    
    # Forward pass
    outputs, attention_maps = model(src, tgt)
    
    # Print some statistics about the output
    print("\n===== FORWARD PASS STATISTICS =====")
    print(f"Output shape: {outputs.shape}")
    print(f"Output stats - Min: {outputs.min().item():.4f}, Max: {outputs.max().item():.4f}, Mean: {outputs.mean().item():.4f}")
    
    # Get complete model state
    model_state = inspect_model_state(model)
    
    # Visualize attention maps
    print("\n===== VISUALIZING ATTENTION MAPS =====")
    
    # Encoder self-attention for first layer, first head
    enc_attn_fig = visualize_attention(
        attention_maps['encoder_attentions'], 
        title='Encoder Self-Attention',
        layer_idx=0, 
        head_idx=0
    )
    
    # Decoder self-attention
    dec_self_attn_fig = visualize_attention(
        attention_maps['decoder_self_attentions'], 
        title='Decoder Self-Attention',
        layer_idx=0, 
        head_idx=0
    )
    
    # Decoder cross-attention
    dec_cross_attn_fig = visualize_attention(
        attention_maps['decoder_cross_attentions'], 
        title='Decoder Cross-Attention',
        layer_idx=0, 
        head_idx=0
    )
    
    # Visualize all encoder heads in the first layer
    all_heads_fig = visualize_all_attention_heads(
        attention_maps['encoder_attentions'],
        layer_idx=0,
        title='Encoder Attention Heads'
    )
    
    return {
        'outputs': outputs,
        'model_state': model_state,
        'attention_maps': attention_maps,
        'figures': {
            'encoder_attention': enc_attn_fig,
            'decoder_self_attention': dec_self_attn_fig,
            'decoder_cross_attention': dec_cross_attn_fig,
            'all_heads': all_heads_fig
        }
    }


def extract_layer_intermediates(model, detailed=False):
    """
    Extract and analyze intermediate matrices from all layers of the model
    
    Args:
        model: The transformer model
        detailed: Whether to print detailed statistics
        
    Returns:
        Dictionary with all intermediate matrices organized by component
    """
    all_intermediates = {}
    
    # Get all intermediates from the model
    model_intermediates = model.get_intermediates()
    
    # Extract encoder layer intermediates
    if 'encoder' in model_intermediates and model_intermediates['encoder']:
        encoder_data = model_intermediates['encoder']
        if 'layers' in encoder_data:
            all_intermediates['encoder_layers'] = {}
            
            for i, layer_data in enumerate(encoder_data['layers']):
                layer_key = f'encoder_layer_{i}'
                all_intermediates['encoder_layers'][layer_key] = layer_data
                
                if detailed:
                    print(f"\nEncoder Layer {i} intermediates:")
                    for tensor_name, tensor in layer_data.items():
                        if isinstance(tensor, torch.Tensor):
                            print(f"  {tensor_name}: Shape {tensor.shape}, Min {tensor.min().item():.4f}, Max {tensor.max().item():.4f}")
    
    # Extract decoder layer intermediates
    if 'decoder' in model_intermediates and model_intermediates['decoder']:
        decoder_data = model_intermediates['decoder']
        if 'layers' in decoder_data:
            all_intermediates['decoder_layers'] = {}
            
            for i, layer_data in enumerate(decoder_data['layers']):
                layer_key = f'decoder_layer_{i}'
                all_intermediates['decoder_layers'][layer_key] = layer_data
                
                if detailed:
                    print(f"\nDecoder Layer {i} intermediates:")
                    for tensor_name, tensor in layer_data.items():
                        if isinstance(tensor, torch.Tensor):
                            print(f"  {tensor_name}: Shape {tensor.shape}, Min {tensor.min().item():.4f}, Max {tensor.max().item():.4f}")
    
    # Extract embedding intermediates
    if 'encoder_embedding' in model_intermediates and model_intermediates['encoder_embedding']:
        all_intermediates['encoder_embedding'] = model_intermediates['encoder_embedding']
    
    if 'decoder_embedding' in model_intermediates and model_intermediates['decoder_embedding']:
        all_intermediates['decoder_embedding'] = model_intermediates['decoder_embedding']
    
    return all_intermediates

def analyze_gradients_by_layer(model, top_k=5):
    """
    Analyze gradients for each layer in the model
    
    Args:
        model: The transformer model
        top_k: Number of parameters with largest gradient norms to show
        
    Returns:
        Dictionary with gradient statistics by layer
    """
    # Get all gradients from the model
    all_gradients = model.get_gradients()
    
    gradient_stats = {}
    
    # Process encoder gradients
    if 'encoder' in all_gradients and all_gradients['encoder']:
        encoder_grads = all_gradients['encoder']
        if 'layers' in encoder_grads:
            gradient_stats['encoder_layers'] = {}
            
            for i, layer_grads in enumerate(encoder_grads['layers']):
                if not layer_grads:
                    continue
                    
                layer_key = f'encoder_layer_{i}'
                layer_stats = {}
                
                # Calculate gradient norms for each parameter
                for param_name, grad in layer_grads.items():
                    if isinstance(grad, torch.Tensor):
                        norm = grad.norm().item()
                        mean = grad.mean().item()
                        std = grad.std().item()
                        layer_stats[param_name] = {'norm': norm, 'mean': mean, 'std': std}
                
                # Sort by norm and keep top_k
                sorted_params = sorted(layer_stats.items(), key=lambda x: x[1]['norm'], reverse=True)
                top_params = sorted_params[:top_k]
                
                print(f"\Encoder Layer {i} - Top {len(top_params)} gradients by norm:")
                for param_name, stats in top_params:
                    print(f"  {param_name}: Norm {stats['norm']:.6f}, Mean {stats['mean']:.6f}, Std {stats['std']:.6f}")
                
                gradient_stats['encoder_layers'][layer_key] = layer_stats
    
    # Process decoder gradients
    if 'decoder' in all_gradients and all_gradients['decoder']:
        decoder_grads = all_gradients['decoder']
        if 'layers' in decoder_grads:
            gradient_stats['decoder_layers'] = {}
            
            for i, layer_grads in enumerate(decoder_grads['layers']):
                if not layer_grads:
                    continue
                    
                layer_key = f'decoder_layer_{i}'
                layer_stats = {}
                
                # Calculate gradient norms for each parameter
                for param_name, grad in layer_grads.items():
                    if isinstance(grad, torch.Tensor):
                        norm = grad.norm().item()
                        mean = grad.mean().item()
                        std = grad.std().item()
                        layer_stats[param_name] = {'norm': norm, 'mean': mean, 'std': std}
                
                # Sort by norm and keep top_k
                sorted_params = sorted(layer_stats.items(), key=lambda x: x[1]['norm'], reverse=True)
                top_params = sorted_params[:top_k]
                
                print(f"\nDecoder Layer {i} - Top {len(top_params)} gradients by norm:")
                for param_name, stats in top_params:
                    print(f"  {param_name}: Norm {stats['norm']:.6f}, Mean {stats['mean']:.6f}, Std {stats['std']:.6f}")
                
                gradient_stats['decoder_layers'][layer_key] = layer_stats
    
    # Process output layer gradients
    if 'output_layer' in all_gradients and all_gradients['output_layer']:
        output_grads = all_gradients['output_layer']
        output_stats = {}
        
        for param_name, grad in output_grads.items():
            if isinstance(grad, torch.Tensor):
                norm = grad.norm().item()
                mean = grad.mean().item()
                std = grad.std().item()
                output_stats[param_name] = {'norm': norm, 'mean': mean, 'std': std}
        
        print("\nOutput Layer gradients:")
        for param_name, stats in output_stats.items():
            print(f"  {param_name}: Norm {stats['norm']:.6f}, Mean {stats['mean']:.6f}, Std {stats['std']:.6f}")
        
        gradient_stats['output_layer'] = output_stats
    
    return gradient_stats

def extract_attention_patterns(model, n_layers=None):
    """
    Extract and organize all attention patterns from the model
    
    Args:
        model: The transformer model
        n_layers: Number of layers to include (None for all)
        
    Returns:
        Dictionary with all attention patterns
    """
    attention_maps = model.get_attention_maps()
    
    attention_patterns = {}
    
    # Process model-level attention maps
    if 'model' in attention_maps:
        model_maps = attention_maps['model']
        
        # Encoder attention
        if 'encoder_attentions' in model_maps:
            encoder_attns = model_maps['encoder_attentions']
            if n_layers is not None:
                encoder_attns = encoder_attns[:n_layers]
            
            attention_patterns['encoder'] = encoder_attns
        
        # Decoder self-attention
        if 'decoder_self_attentions' in model_maps:
            decoder_self_attns = model_maps['decoder_self_attentions']
            if n_layers is not None:
                decoder_self_attns = decoder_self_attns[:n_layers]
            
            attention_patterns['decoder_self'] = decoder_self_attns
        
        # Decoder cross-attention
        if 'decoder_cross_attentions' in model_maps:
            decoder_cross_attns = model_maps['decoder_cross_attentions']
            if n_layers is not None:
                decoder_cross_attns = decoder_cross_attns[:n_layers]
            
            attention_patterns['decoder_cross'] = decoder_cross_attns
    
    return attention_patterns

def compute_attention_stats(attention_patterns):
    """
    Compute statistics for attention patterns
    
    Args:
        attention_patterns: Dictionary with attention patterns
        
    Returns:
        Dictionary with attention statistics
    """
    stats = {}
    
    for attn_type, layers in attention_patterns.items():
        layer_stats = []
        
        for layer_idx, layer_attn in enumerate(layers):
            # Get attention tensor shape [batch_size, num_heads, seq_len_q, seq_len_k]
            batch_size, num_heads, seq_len_q, seq_len_k = layer_attn.shape
            
            # Compute statistics per head
            head_stats = []
            for head_idx in range(num_heads):
                head_attn = layer_attn[:, head_idx, :, :]
                
                # Compute entropy (measure of attention focus)
                # Lower entropy means more focused attention
                entropy = -torch.sum(head_attn * torch.log(head_attn + 1e-10), dim=-1).mean().item()
                
                # Compute sparsity (percentage of attention weights below threshold)
                threshold = 0.01
                sparsity = (head_attn < threshold).float().mean().item()
                
                # Compute attention to diagonals (for self-attention, measure of local focus)
                if seq_len_q == seq_len_k:
                    diag_indices = torch.arange(seq_len_q)
                    diag_attn = head_attn[:, diag_indices, diag_indices].mean().item()
                else:
                    diag_attn = None
                
                head_stats.append({
                    'entropy': entropy,
                    'sparsity': sparsity,
                    'diagonal_attention': diag_attn
                })
            
            layer_stats.append(head_stats)
        
        stats[attn_type] = layer_stats
    
    return stats

def visualize_attention_patterns(attention_patterns, layer_idx=0):
    """
    Create visualizations for different types of attention patterns
    
    Args:
        attention_patterns: Dictionary with attention patterns
        layer_idx: Which layer to visualize
        
    Returns:
        Dictionary with matplotlib figures
    """
    figures = {}
    
    for attn_type, layers in attention_patterns.items():
        if layer_idx >= len(layers):
            continue
            
        attn = layers[layer_idx]
        
        # Visualize all heads in this layer
        fig = visualize_all_attention_heads(
            attn, 
            title=f"{attn_type.replace('_', ' ').title()} Attention"
        )
        
        figures[f"{attn_type}_layer{layer_idx}"] = fig
    
    return figures

def train_with_gradient_tracking(model, train_loader, num_epochs=1, lr=0.0001):
    """Train the model while tracking gradients"""
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Create loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is padding
    
    # Create gradient tracker
    gradient_tracker = GradientTracker(model)
    
    # Store intermediate attention patterns and gradients
    attention_history = []
    gradient_history = []
    loss_history = []
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        epoch_losses = []
        
        for batch_idx, (src, tgt) in enumerate(train_loader):
            # Forward pass
            logits, attention_maps = model(src, tgt[:, :-1])  # shift right for teacher forcing
            
            # Calculate loss
            loss = criterion(
                logits.contiguous().view(-1, logits.size(-1)), 
                tgt[:, 1:].contiguous().view(-1)
            )
            
            # Reset gradients
            optimizer.zero_grad()
            gradient_tracker.reset()
            
            # Backward pass
            loss.backward()
            
            # Capture gradients
            gradient_tracker.capture()
            
            # Store attention patterns and gradients at intervals
            if batch_idx % 10 == 0:
                # Extract and store attention patterns
                attention_patterns = extract_attention_patterns(model)
                attention_history.append({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'patterns': copy.deepcopy(attention_patterns)
                })
                
                # Extract and store gradients
                all_gradients = model.get_gradients()
                gradient_history.append({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'gradients': copy.deepcopy(all_gradients)
                })
            
            # Update weights
            optimizer.step()
            
            # Store loss
            current_loss = loss.item()
            total_loss += current_loss
            epoch_losses.append(current_loss)
            loss_history.append({
                'epoch': epoch,
                'batch': batch_idx,
                'loss': current_loss
            })
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {current_loss:.4f}")
                
                # Plot gradient norms for last 20 iterations
                if batch_idx > 0 and batch_idx % 50 == 0:
                    gradient_tracker.plot_gradient_norms(top_k=5, last_n=20)
                    plt.show()
                    
                    # Analyze gradients by layer
                    analyze_gradients_by_layer(model, top_k=3)
        
        # Compute epoch statistics
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed, Average Loss: {avg_loss:.4f}")
        
        # Plot loss curve for this epoch
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_losses)
        plt.title(f"Loss Curve - Epoch {epoch+1}")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(f"epoch_{epoch+1}_loss.png")
        plt.close()
        
        # Examine attention patterns at the end of each epoch
        print(f"\nExamining attention patterns at end of epoch {epoch+1}:")
        attention_patterns = extract_attention_patterns(model)
        stats = compute_attention_stats(attention_patterns)
        
        # Print some stats about encoder attention
        if 'encoder' in stats:
            for layer_idx, layer_stats in enumerate(stats['encoder']):
                print(f"  Encoder Layer {layer_idx+1}:")
                for head_idx, head_stats in enumerate(layer_stats):
                    entropy = head_stats.get('entropy', float('nan'))
                    sparsity = head_stats.get('sparsity', float('nan'))
                    print(f"    Head {head_idx+1}: Entropy={entropy:.4f}, Sparsity={sparsity:.4f}")
        
        # Create visualization of attention patterns
        figs = visualize_attention_patterns(attention_patterns)
        for name, fig in figs.items():
            fig.savefig(f"epoch_{epoch+1}_{name}.png")
            plt.close(fig)
    
    # Clean up hooks
    gradient_tracker.remove_hooks()
    
    # Final statistics and visualizations
    print("\n===== TRAINING COMPLETED =====")
    
    # Plot overall loss curve
    plt.figure(figsize=(10, 6))
    losses = [entry['loss'] for entry in loss_history]
    plt.plot(losses)
    plt.title("Loss Curve - Full Training")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("full_training_loss.png")
    plt.close()
    
    # Final gradient analysis
    print("\nFinal Gradient Analysis:")
    analyze_gradients_by_layer(model)
    
    # Final model inspection
    inspect_model_state(model)
    
    return {
        'gradient_tracker': gradient_tracker,
        'attention_history': attention_history,
        'gradient_history': gradient_history,
        'loss_history': loss_history
    }