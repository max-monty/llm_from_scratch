import torch
import torch.nn.functional as F
import tiktoken
from gpt.dataset import GPTDataset

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        x = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def load_dataset(cfg, filepath): 
    with open(filepath, "r", encoding="utf-8") as file:
        raw_text = file.read()
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(raw_text, tokenizer, context_size=cfg["context_len"], stride=1)
    dataloader = dataset.load_data(batch_size=cfg["batch_size"], shuffle=True)
    iter_dataloader = iter(dataloader)
    return iter_dataloader, tokenizer

def create_dataloader(cfg, data, tokenizer):
    dataset = GPTDataset(data, tokenizer, context_size=cfg["context_len"], stride=1)
    dataloader = dataset.load_data(batch_size=cfg["batch_size"], shuffle=True)
    iter_dataloader = iter(dataloader)
    return iter_dataloader

def create_dataloader_batch(cfg, train_data, val_data, tokenizer):
    train_dataloader = create_dataloader(cfg, train_data, tokenizer)
    val_dataloader = create_dataloader(cfg, val_data, tokenizer)
    return train_dataloader, val_dataloader


def decode_batch(batch, tokenizer): 
    out = []
    for i in range(batch.shape[0]):
        out.append(tokenizer.decode(batch[i].tolist()))
    return out

def calculate_loss(logits, targets):
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    targets = targets.view(B*T)
    loss = F.cross_entropy(logits, targets)
    return loss

def calc_loss_batch(input, target, model, device):
    input, target = input.to(device), target.to(device)
    logits = model(input)
    loss = F.cross_entropy(logits.flatten(0, 1), target.flatten())
    return loss

def calc_loss_loader(dataloader, model, device, n_batches=None):
    total_loss = 0.0
    if len(dataloader) == 0:
        return float("nan")
    elif n_batches is None:
        n_batches = len(dataloader)
    else:
        n_batches = min(n_batches, len(dataloader))
    
    for i, (x, y) in enumerate(dataloader):
        if i == n_batches:
            break
        loss = calc_loss_batch(x, y, model, device)
        total_loss += loss.item()
    
    return total_loss / n_batches