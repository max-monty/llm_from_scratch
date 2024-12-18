import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(cfg["emb_dim"]))
        self.bias = nn.Parameter(torch.zeros(cfg["emb_dim"]))
        self.eps = 1e-5
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normalized = (x - mean) / (std + self.eps)
        return normalized * self.weight + self.bias

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_len, embed_size)
        
    def forward(self, x):
        tok_embed = self.token_embedding(x)
        pos_embed = self.position_embedding(torch.arange(x.shape[1], device=x.device))
        return tok_embed + pos_embed

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, n_heads, drop_rate, qkv_bias):
        super().__init__()
        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        self.q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(drop_rate)
        self.out_proj = nn.Linear(d_out, d_out)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
        
    def forward(self, x):
        B, T, C = x.shape
        Q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attention_scores = Q @ K.transpose(-2, -1)
        mask = self.mask[:T, :T]
        attention_scores = attention_scores.masked_fill(mask.bool(), -torch.inf)
        attention_weights = torch.softmax(attention_scores / (self.head_dim ** 0.5), dim=-1)
        attention_weights = self.dropout(attention_weights)
        context_vectors = attention_weights @ V
        context_vectors = context_vectors.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(context_vectors)

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
        
    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_len"],
            n_heads=cfg["n_heads"],
            drop_rate=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = MLP(cfg)
        self.norm_1 = LayerNorm(cfg)
        self.norm_2 = LayerNorm(cfg)
        self.dropout = nn.Dropout(cfg["drop_rate"])
        
    def forward(self, x):
        short_cut = x
        x = self.norm_1(x)
        x = self.att(x)
        x = self.dropout(x)
        x = x + short_cut

        short_cut = x
        x = self.norm_2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + short_cut
        return x

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding_layer = EmbeddingLayer(cfg["vocab_size"], cfg["emb_dim"], cfg["context_len"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.ln_f = LayerNorm(cfg)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        
    def forward(self, x):
        x = self.drop_emb(self.embedding_layer(x))
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.out_head(x)