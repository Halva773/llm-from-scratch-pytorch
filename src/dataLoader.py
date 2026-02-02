from torch.utils.data import Dataset
import torch


class GetData(Dataset):
    def __init__(self, data: list, seq_len: int, device: str = 'cpu'):
        self.data = data
        self.seq_len = seq_len
        self.device = device


    def __len__(self):
        return len(self.data) - self.seq_len
    

    def __getitem__(self, idx: int):
        x = torch.tensor(self.data[idx:idx + self.seq_len ], device=self.device)
        y = torch.tensor(self.data[idx + 1:idx + self.seq_len + 1], device=self.device)
        return x, y