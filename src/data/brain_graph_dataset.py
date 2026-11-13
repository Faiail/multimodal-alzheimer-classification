from torch.utils.data import Dataset as TorchDataset, DataLoader
from .utils import BatchKeys
import pandas as pd
import torch
from torch_geometric.loader.dataloader import Collater


class BrainGraphDataset(TorchDataset):
    def __init__(
        self,
        graph_root: str,
        fold_root: str,
        split: str = "train",
        fold_id: int = None,
    ):
        super().__init__()
        self.graph_root = graph_root
        fold = f"folds/{fold_id}" if fold_id is not None else ""
        self.dataset = pd.read_csv(f"{fold_root}/{fold}/{split}.csv")
        self.dataset[BatchKeys.SAFETENSOR_PTH] = self.dataset[
            BatchKeys.SAFETENSOR_PTH
        ].map(lambda x: x.replace(".safetensors", ".pt"))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        file_pth = f"{self.graph_root}/{row[BatchKeys.SAFETENSOR_PTH]}"
        label = row[BatchKeys.LABEL]
        graph = torch.load(file_pth, weights_only=False)
        return {BatchKeys.GRAPH: graph, BatchKeys.GT: label}


class BrainGraphCollater(Collater):
    def __call__(self, batch: list[dict]):
        graphs = [item.get(BatchKeys.GRAPH) for item in batch]
        collated_graphs = super().__call__(graphs)
        gts = torch.as_tensor([item.get(BatchKeys.GT) for item in batch])
        return {BatchKeys.GRAPH: collated_graphs, BatchKeys.GT: gts}


class BrainGraphLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=None,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor=None,
        persistent_workers=False,
        pin_memory_device="",
        follow_batch=None,
        exclude_keys=None,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=BrainGraphCollater(
                dataset=dataset,
                follow_batch=follow_batch,
                exclude_keys=exclude_keys,
            ),
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )
