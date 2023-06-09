from datasets.vulnerability_dataset import VulnerabilityDataset
from datasets.parameter import PARAMS

dataset = VulnerabilityDataset(**PARAMS)

for i, data in enumerate(dataset):
    print(data)
