from datasets.vulnerability_dataset import VulnerabilityDataset

PARAMS = {
    "dataset_dir": "data/reveal/",
    "cache_dir": "REVEAL",
    "c_in_dot": True,
    "encoding_params": {
        "vector_size": 128,
        "model_file": "model/transformer_model.chkpt",
        "training_files": [("data/reveal", True)]
    },
    "max_seq_len": 4096,
    "name": "REVEAL",
    "overwrite_cache": False
}

dataset = VulnerabilityDataset(**PARAMS)

dataset.get_input_size()
dataset.get_edge_size()
