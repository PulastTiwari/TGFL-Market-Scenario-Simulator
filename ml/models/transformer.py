"""
Lightweight Transformer model for time series generation
Optimized for CPU training and small memory footprint
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import numpy as np

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """Lightweight multi-head attention"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear transformations and reshape to (batch_size, n_heads, seq_len, d_k)
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        
        return self.w_o(context)

class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Single transformer encoder block"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class LightweightTransformer(nn.Module):
    """
    Lightweight Transformer for time series generation
    Optimized for CPU training with <1M parameters
    """
    
    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        d_ff_multiplier: int = 2
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # For continuous time series, we'll use a linear projection instead of embedding
        self.input_projection = nn.Linear(1, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks
        d_ff = d_model * d_ff_multiplier
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal mask for autoregressive generation"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, seq_len)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, 1) or (batch_size, seq_len)
            mask: Optional attention mask
            
        Returns:
            Output predictions of shape (batch_size, seq_len, 1)
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Ensure input has correct shape
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # Add feature dimension
            
        # Project input to model dimension
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # Create causal mask if not provided
        if mask is None:
            mask = self.create_causal_mask(seq_len).to(x.device)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Project to output
        output = self.output_projection(x)  # (batch_size, seq_len, 1)
        
        return output
    
    def generate(
        self, 
        context: torch.Tensor, 
        length: int,
        temperature: float = 1.0,
        top_k: int = 0
    ) -> torch.Tensor:
        """
        Generate sequences autoregressively
        
        Args:
            context: Initial context of shape (1, context_len, 1)
            length: Number of steps to generate
            temperature: Sampling temperature (1.0 = no change, <1.0 = more conservative)
            top_k: If >0, only sample from top-k predictions
            
        Returns:
            Generated sequence of shape (1, context_len + length, 1)
        """
        self.eval()
        
        with torch.no_grad():
            generated = context.clone()
            
            for _ in range(length):
                # Forward pass on current sequence
                if generated.size(1) > self.max_seq_len:
                    # Truncate to max sequence length
                    input_seq = generated[:, -self.max_seq_len:, :]
                else:
                    input_seq = generated
                
                logits = self.forward(input_seq)
                
                # Get prediction for the last position
                next_token_logits = logits[:, -1, :] / temperature
                
                if top_k > 0:
                    # Top-k sampling
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    # Create a distribution with only top-k values
                    probs = torch.zeros_like(next_token_logits)
                    probs.scatter_(1, top_k_indices, F.softmax(top_k_logits, dim=-1))
                else:
                    probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                if temperature == 0.0:
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)
                else:
                    next_token = torch.multinomial(probs, 1)
                
                # Convert back to continuous value (this is a simplification)
                # In practice, you might want to use a continuous sampling strategy
                next_value = next_token.float().unsqueeze(-1)  # Shape: (1, 1, 1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_value], dim=1)
        
        return generated
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_small_transformer() -> LightweightTransformer:
    """Create a small transformer model with <1M parameters"""
    return LightweightTransformer(
        vocab_size=1000,
        d_model=64,
        n_heads=4,
        n_layers=3,
        max_seq_len=256,
        dropout=0.1,
        d_ff_multiplier=2
    )

def create_tiny_transformer() -> LightweightTransformer:
    """Create a tiny transformer model with <500K parameters"""
    return LightweightTransformer(
        vocab_size=1000,
        d_model=48,
        n_heads=4,
        n_layers=2,
        max_seq_len=128,
        dropout=0.05,
        d_ff_multiplier=2
    )

# Example usage and testing
if __name__ == "__main__":
    # Test model creation
    model = create_small_transformer()
    print(f"Small model parameters: {model.count_parameters():,}")
    
    tiny_model = create_tiny_transformer()
    print(f"Tiny model parameters: {tiny_model.count_parameters():,}")
    
    # Test forward pass
    batch_size, seq_len = 8, 64
    x = torch.randn(batch_size, seq_len, 1)
    
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output.shape}")
        
    # Test generation
    context = torch.randn(1, 10, 1)
    generated = model.generate(context, length=20)
    print(f"Generated sequence shape: {generated.shape}")