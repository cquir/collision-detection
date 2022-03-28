import torch
import os
import pandas as pd

# this rly shouldn't be hardcoded but w/e being lazy
cols = [
    "pos0_x",
    "pos0_y",
    "pos0_z",
    "pos1_x",
    "pos1_y",
    "pos1_z",
    "quat0_a",
    "quat0_b",
    "quat0_c",
    "quat0_d",
    "quat1_a",
    "quat1_b",
    "quat1_c",
    "quat1_d",
]

class ColliderDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir : str):
        self.data_dir = data_dir
        self.names = [ e for e in os.listdir(data_dir) if e.endswith(".parquet") ]

        df = pd.read_parquet(f"{self.data_dir}/{self.names[0]}")

        X = torch.tensor(df[cols].values, dtype=torch.float32)
        Y = torch.tensor(df["collides"].values, dtype=torch.float32)

        self.data = (X, Y)

    def __len__(self):
        # return len(self.names)
        return len(self.data[0])

    def __getitem__(self, i : int):

        # df = pd.read_parquet(f"{self.data_dir}/{self.names[i]}")

        # X = torch.tensor(df[cols[:-1]].values)
        # Y = torch.tensor(df["collides"].values)

        # print(X.shape)

        # exit()

        # return X, Y
        return (self.data[0][i], self.data[1][i])
    
    

if __name__ == "__main__":

    dataset = ColliderDataset("./data/datasets/smoketest")
    
    for e in dataset:
        print(e)