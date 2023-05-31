from datasets.vulnerability_dataset import VulnerabilityDataset

PARAMS = {
    "dataset_dir": "cache/reveal_cpg/",
    "cache_dir": "REVEAL",
    "c_in_dot": True,
    "encoding_params": {
        "vector_size": 128,
        "training_files": [("cache/reveal_cpg", False)]
    },
    "max_seq_len": 4096,
    "name": "REVEAL",
    "overwrite_cache": False
}

dataset = VulnerabilityDataset(**PARAMS)

dataset.get_input_size()
dataset.get_edge_size()
