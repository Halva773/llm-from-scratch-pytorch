from torch.utils.data import Dataset
import torch


class GetData(Dataset):
    def __init__(self, data: list[int], seq_len: int):
        self.data = torch.tensor(data, dtype=torch.long)
        self.seq_len = seq_len


    def __len__(self):
        return len(self.data) - self.seq_len - 1
    

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y
