from datasets.vulnerability_dataset import VulnerabilityDataset
from datasets.parameter import PARAMS
from tqdm import tqdm
import torch
import os

dataset = VulnerabilityDataset(**PARAMS)

if not os.path.exists("data/processed"):
    os.makedirs("data/processed")

for i, data in enumerate(tqdm(dataset)):
    torch.save(data, f'data/processed/{data.idx}_cpg.pt')
    print(data)
