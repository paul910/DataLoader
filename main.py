from datasets.vulnerability_dataset import VulnerabilityDataset
from datasets.parameter import PARAMS
from tqdm import tqdm

dataset = VulnerabilityDataset(**PARAMS)

for i, data in enumerate(tqdm(dataset)):
    print(data)
