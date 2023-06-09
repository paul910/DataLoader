import glob
import gzip
import json
import os
import pickle
import random
import traceback
import pathlib
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Subset
from torch_geometric.data import Data
from tqdm import tqdm

from torch.utils.data import Dataset
from .transformer import TransformerBuilder
from .vulnerability_dataset_utils import read_dot, from_networkx_multi, ASTEncoder


class VulnerabilityDataset(Dataset):
    def __init__(self, **params):
        self.params = params
        self.params["max_num_nodes"] = params.get("max_num_nodes", 1000)
        self.dataset_dir = params["dataset_dir"]
        self.cache_dir = os.path.join("cache", params["cache_dir"])
        self.overwrite_cache = params["overwrite_cache"]
        self.bounds = params.get("encode_bounds", False)
        self.max_seq_len = params.get("max_seq_len", None)
               
        os.makedirs('test/')

        # Now create a file in this new directory.
        with open('test/file.txt', 'w') as file:
            file.write('Hello, world!')
        

        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)

        self.graph_files = list(glob.glob(f"{self.dataset_dir}/**/*.cpg"))
        np.random.shuffle(self.graph_files)

        print("Building text model")
        self.text_model = TransformerBuilder(params["encoding_params"])
        self.text_model.build_model()
        print("Loading files")
        self._load_files()
        print("Building AST encoder")
        self.astencoder = ASTEncoder({
            "files": params["path_to_cpgs"],
            "overwrite_cache": params["overwrite_cache"]
        })
        self._build_sets()

    def _get_meta_cache_path(self):
        return os.path.join(self.cache_dir, "meta.pkl.gz")

    def _zip_path_for(self, cpg_path):
        _, filename = os.path.split(cpg_path)
        filename += ".pt.gz"
        return os.path.join(self.cache_dir, filename)

    def get_edge_size(self):
        return None

    def _check_seq_len(self, graph):
        return all(
            len(self.text_model._tokenize(
                graph.nodes[node_id]["enclosing"]
            )) <= self.max_seq_len
            for node_id in graph
        )

    def _load_files(self):
        if os.path.isfile(self._get_meta_cache_path()) and not self.overwrite_cache:
            with gzip.open(self._get_meta_cache_path(), "r") as f:
                data = pickle.load(f)
                self.graph_files = data[0]
                self.class_indices = data[1]
                self.asttypes = data[2]
                return
        failed = 0
        success = []
        self.asttypes = defaultdict(int)
        self.class_indices = defaultdict(list)
        progress_bar = tqdm(self.graph_files)
        for path in progress_bar:
            try:
                f = open(path, "r")
                graph = read_dot(f)
                if len(graph) > self.params["max_num_nodes"] or len(graph) < 10:
                    print(f"Skipping graph with {len(graph)} nodes")
                    continue
                if self.max_seq_len is not None and not self._check_seq_len(graph):
                    print("Skipping graph with too long sequence")
                    continue
                label = self._label_for(path)

                graph.graph["label"] = label
                self._get_meta(graph)

                self.class_indices[label].append(len(success))
                success.append(path)
            except Exception as e:
                print(f"Failed {path} with {repr(e)}")
                print(traceback.format_exc())
                failed += 1
            finally:
                f.close()
            progress_bar.set_description(f"failed: {failed} success: {len(success)}")
        self.graph_files = success
        print(
            f"negative samples: {len(self.class_indices[0])}, positive samples:{len(self.class_indices[1])} failed:{failed}")
        with gzip.open(self._get_meta_cache_path(), "w") as f:
            pickle.dump((self.graph_files, self.class_indices, self.asttypes), f)

    def _label_for(self, path):
        if "/benign/" in path or "/mal2/" in path:
            # paths dataset
            return 0 if "/benign/" in path else 1
        else:
            return int(os.path.split(path)[1].split("_")[-1].split(".")[0])

    def _get_meta(self, graph):
        for node in graph:
            node = graph.nodes[node]
            self.asttypes[node["label"]] += 1

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        path = self.graph_files[idx]
        zip_path = self._zip_path_for(path)
        if os.path.isfile(zip_path) and not self.overwrite_cache:
            with gzip.open(zip_path, "r") as f:
                try:
                    data = Data.from_dict(pickle.load(f))
                    xs = [data.astenc, data.codeenc]
                    if self.bounds:
                        xs.extend([data.lowerBound, data.upperBound])
                    datanew = Data(idx=idx, y=data.y,
                                   x=torch.cat(xs, dim=-1).float(), edge_index=data.edge_index)
                    datanew.num_nodes = data.astenc.shape[0]
                    return datanew
                except Exception as e:
                    print(f"cached {path} failed with {e}, recomputing")

        with open(path, "r") as f:
            graph = read_dot(f)
            label = self._label_for(path)
            graph.graph["label"] = label

            data = self.encode(graph)

            with gzip.open(zip_path, "w") as g:
                pickle.dump(data.to_dict(), g)

            xs = [data.astenc, data.codeenc]
            if self.bounds:
                xs.extend([data.lowerBound, data.upperBound])
            data.x = torch.cat(xs, dim=-1)

            datanew = Data(idx=idx, y=data.y,
                           x=data.x.float(), edge_index=data.edge_index)
            datanew.num_nodes = data.astenc.shape[0]

            return datanew

    def encode(self, graph):

        for node_id in graph:
            node = graph.nodes[node_id]
            if self.bounds == "random":
                node["upperBound"] = 0 if random.random() < 0.5 else 1
                node["lowerBound"] = 0 if random.random() < 0.5 else 1
            elif type(self.bounds) is bool and self.bounds:
                node["upperBound"] = 0 if node["upperBound"] in ["EMPTY_STRING", "unchecked"] else 1
                node["lowerBound"] = 0 if node["lowerBound"] in ["EMPTY_STRING", "unchecked"] else 1

            node["ast"] = node["label"]
            node["lines"] = node["location"]
            node["code"] = node["enclosing"]
            continue

        for node in graph:
            asttype = self.astencoder._clean(graph.nodes[node]["ast"])
            if not asttype in self.ast_dict:
                asttype = "UNKNOWN"
            graph.nodes[node]["astenc"] = self.ast_dict[asttype]
            graph.nodes[node]["codeenc"] = self.text_model.get_embedding(graph.nodes[node]["code"])
            if self.bounds:
                graph.nodes[node]["upperBound"] = np.asarray([graph.nodes[node]["upperBound"]], dtype=np.float32)
                graph.nodes[node]["lowerBound"] = np.asarray([graph.nodes[node]["lowerBound"]], dtype=np.float32)
        try:
            torch_graph = from_networkx_multi(graph)
            torch_graph.y = graph.graph["label"]
            return torch_graph
        except Exception as e:
            print(f"failed with {repr(e)}")
            print(traceback.format_exc())

    def _build_sets(self):
        print("building sets")

        self.asttypes = self.astencoder.get_asttypes()
        asttypes = list(self.asttypes.keys())
        asttypes.append("UNKNOWN")
        self.astenc = OneHotEncoder(handle_unknown="ignore", sparse=False)
        self.astenc.fit(np.array(asttypes).reshape(-1, 1))
        self.ast_dict = {asttype: np.squeeze(self.astenc.transform(np.array(asttype).reshape(1, -1)))
                         for asttype in asttypes}
        print("encoder generated")

    def get_input_size(self):
        bounds_size = 2 if self.bounds else 0
        return len(self.astenc.categories_[0]) + self.text_model.vector_size + bounds_size

    def get_params(self):
        return self.params

    def get_graph_identifiers(self) -> List:
        return self.graph_files

    def get_subset_of(self, clas):
        return Subset(self, self.class_indices[clas])

    def get_classes(self):
        return self.class_indices.keys()

    def get_cwes(self):
        cve2cwe_path = os.path.join(self.dataset_dir, "cve2cwe.json")
        cves_path = os.path.join(self.dataset_dir, "patchdb-cves.csv")
        if not os.path.exists(cve2cwe_path) or not os.path.exists(cves_path):
            raise Exception("Dataset does not have cve information")

        with open(cve2cwe_path, "r") as f:
            cve2cwe = json.load(f)

        with open(cves_path, "r") as f:
            cves = f.readlines()[1:]

        cves = [ \
            {"cve": cve[0], "commit": cve[1], "project": os.path.split(cve[2])[1].strip()[:-len(".git")],
             "cwes": cve2cwe.get(cve[0])} \
            for cve in (row.split(",") for row in cves) \
            ]

        graph_ids = [ \
            {"project": id_parts[0], "commit": id_parts[1], "graph_id": graph_id} \
            for (id_parts, graph_id) in
            ((os.path.split(graph_id)[1].split("_"), graph_id) for graph_id in self.get_graph_identifiers()) \
            if int(id_parts[-1].split(".")[0]) == 1
        ]

        project_commit_to_cwes = {(cve["project"], cve["commit"]): cve["cwes"] for cve in cves}
        cwes = defaultdict(list)

        for graph_id in graph_ids:
            this_cwes = project_commit_to_cwes.get((graph_id["project"], graph_id["commit"]))
            if this_cwes is None or len(this_cwes) == 0:
                continue
            for cwe in this_cwes:
                cwes[cwe].append(graph_id["graph_id"])

        return cwes

    def split(self, factor=0.8) -> Tuple[Subset, Subset]:
        assert factor > 0
        assert factor < 1

        number_graphs = len(self)

        all_indices = list(range(0, number_graphs))
        np.random.shuffle(all_indices)

        train_indices = all_indices[:int(number_graphs * factor)]
        test_indices = all_indices[int(number_graphs * factor):]

        minimum_train_graphs = max(factor - 0.1, 0.05)
        minimum_test_graphs = max(1 - factor - 0.1, 0.05)

        assert len(train_indices) > minimum_train_graphs * number_graphs
        assert len(test_indices) > minimum_test_graphs * number_graphs
        assert set(train_indices).isdisjoint(set(test_indices))

        train_dataset = Subset(self, train_indices)
        test_dataset = Subset(self, test_indices)

        return train_dataset, test_dataset
