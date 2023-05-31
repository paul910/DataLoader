from typing import List

from torch.utils.data import Subset

from . import ClassificationDataset


def split_dataset_by_ids(dataset: ClassificationDataset, idss: List[List]) -> List[Subset]:
    id_index_map = {id: index for (index, id) in enumerate(dataset.get_graph_identifiers())}

    subsets = list()
    for ids in idss:
        indices = [id_index_map[id] for id in ids]
        subsets.append(Subset(dataset, indices))
    return subsets
