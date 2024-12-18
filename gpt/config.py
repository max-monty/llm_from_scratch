GPT_CONFIG_SMALL = {
    "vocab_size": 50257,    # Vocabulary size
    "n_layers": 12,         # Number of layers
    "n_heads": 12,          # Number of attention heads
    "emb_dim": 768,         # Embedding dimension
    "context_len": 1024,    # Context length
    "drop_rate": 0.1,       # Dropout rate 
    "qkv_bias": False,      # Query-Key-Value bias
    "batch_size": 32,        # Batch size
}

GPT_CONFIG_MEDIUM = {
    "vocab_size": 50257,
    "n_layers": 24,
    "n_heads": 16,
    "emb_dim": 1024,
    "context_len": 1024,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "batch_size": 32,
}

GPT_CONFIG_LARGE = {
    "vocab_size": 50257,
    "n_layers": 36,
    "n_heads": 20,
    "emb_dim": 1280,
    "context_len": 1024,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "batch_size": 32,
}

GPT_CONFIG_XL = {
    "vocab_size": 50257,
    "n_layers": 48,
    "n_heads": 25,
    "emb_dim": 1600,
    "context_len": 1024,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "batch_size": 32,
}

GPT_CONFIG_TEST = {
    "vocab_size": 50257,
    "n_layers": 3,
    "n_heads": 3,
    "emb_dim": 768,
    "context_len": 256,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "batch_size": 32,
}
