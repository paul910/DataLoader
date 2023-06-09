from datasets.vulnerability_dataset import VulnerabilityDataset
from datasets.parameter import PARAMS
from tqdm import tqdm
import torch

dataset = VulnerabilityDataset(**PARAMS)

for i, data in enumerate(tqdm(dataset)):
    torch.save(data, f'{data.idx}_cpg.pt')
    print(data)
