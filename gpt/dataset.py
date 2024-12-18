import torch
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, context_size=4, stride=1):
        self.x = []
        self.y = []
        enc_txt = tokenizer.encode(text)
        for i in range(0, len(enc_txt) - context_size, stride):
            x = enc_txt[i:i+context_size]
            y = enc_txt[i+1:i+context_size+1]
            self.x.append(torch.tensor(x))
            self.y.append(torch.tensor(y))
        self.x = torch.stack(self.x)
        self.y = torch.stack(self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def load_data(self, batch_size=32, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=0)