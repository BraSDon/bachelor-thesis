import os
from typing import Optional, List

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset, DistributedSampler

from .chunker import SequentialChunker, StepChunker
from .plot_helper import PlotHelper, get_label_to_count


class DataDistributer:
    """
    Wrapper of DataLoader.
    Operates in two different modes:
        1. Global: All ranks have the same DataLoader.
        2. Local: Each rank has its own DataLoader handling a disjoint partition of the full dataset.

    - Removing samples if the dataset is not divisible by the number of ranks.

    Args:
        full_dataset: The full dataset.
        chunker: The chunker to use for splitting the dataset. Defaults to SequentialChunker.
        sorted_indices: If not None, the indices to use for splitting the dataset sequentially.
        initial_shuffle: If True, shuffle the dataset before splitting it.
        global_: If True, all ranks have the same DataLoader.
        verbose: If True, print information about the class and create a histogram of the labels.

        the rest: Same as DataLoader.
    """
    _data_loader = None
    _full_dataset = None
    __subsets = None
    __data_loaders = None
    __rank = None
    __world_size = None
    __chunker = None
    __sorted_indices = None
    __initial_shuffle = False
    __generator = None

    def __init__(self, full_dataset, chunker=None, sorted_indices=None, initial_shuffle=False,
                 global_=False, verbose=True,
                 batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = None, sampler=None,
                 num_workers: int = 0, collate_fn=None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn=None,
                 multiprocessing_context=None, generator=None,
                 prefetch_factor: int = 2,
                 persistent_workers: bool = False,
                 pin_memory_device: str = ""):
        self._full_dataset = full_dataset
        self._global_ = global_
        self.__rank = dist.get_rank()
        self.__world_size = dist.get_world_size()
        self.__sorted_indices = sorted_indices
        self.__initial_shuffle = initial_shuffle

        # Attributes that need to be set before calling __get_subsets.
        self.__chunker = chunker if chunker else SequentialChunker()
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        self.__generator = generator

        # Global case: pre_step_global
        # - Use DistributedSampler to split the dataset.
        if global_:
            if not (shuffle and initial_shuffle):
                raise ValueError("global_ option requires shuffle and initial_shuffle to be True.")
            if sorted_indices is not None:
                raise ValueError("global_ option requires sorted_indices to be None.")

            indices = torch.randperm(len(self._full_dataset), generator=generator)
            shuffled_dataset = Subset(self._full_dataset, indices)
            distributed_sampler = DistributedSampler(shuffled_dataset, shuffle=shuffle)
            self._data_loader = DataLoader(shuffled_dataset, batch_size=batch_size, shuffle=False,
                                           sampler=distributed_sampler, num_workers=num_workers,
                                           collate_fn=collate_fn, pin_memory=pin_memory,
                                           drop_last=drop_last, timeout=timeout,
                                           worker_init_fn=worker_init_fn,
                                           multiprocessing_context=multiprocessing_context,
                                           generator=generator, prefetch_factor=prefetch_factor,
                                           persistent_workers=persistent_workers,
                                           )
        else:
            self.__subsets = self.__get_subsets(sorted_indices, initial_shuffle)
            self.__data_loaders = self.__get_data_loaders(batch_size=batch_size,
                                                          shuffle=shuffle, sampler=sampler,
                                                          num_workers=num_workers,
                                                          collate_fn=collate_fn,
                                                          pin_memory=pin_memory, drop_last=drop_last,
                                                          timeout=timeout,
                                                          worker_init_fn=worker_init_fn,
                                                          multiprocessing_context=multiprocessing_context,
                                                          generator=generator,
                                                          prefetch_factor=prefetch_factor,
                                                          persistent_workers=persistent_workers,
                                                          )
        if verbose:
            self.print0(f"DataDistributer attributes:\n{self.__repr__()}")
            # Do not create histogram in global case.
            if global_: return
            self.print0("WARNING: Being verbose creates significant "
                        "overhead (up to 20min on ImageNet). Deactivate if not needed!")
            try:
                self._full_dataset.stop_recording()
                self.__create_and_store_histogram()
                self._full_dataset.start_recording()
            except AttributeError:
                self.__create_and_store_histogram()

    def __get_subsets(self, indices, initial_shuffle=False) -> List[Subset]:
        """
        Evenly split the dataset into subsets for each rank.
        Splitting is done according to chunker.
        """
        # Trim dataset to be divisible by world size.
        if len(self._full_dataset) % self.__world_size != 0:
            new_length = len(self._full_dataset) - (len(self._full_dataset) % self.__world_size)
            self._full_dataset = Subset(self._full_dataset, range(new_length))

        if indices is None:
            if initial_shuffle:
                assert self.__generator is not None
                indices = torch.randperm(len(self._full_dataset),
                                         generator=self.__generator).tolist()
            else:
                indices = list(range(len(self._full_dataset)))
        elif isinstance(self.__chunker, StepChunker):
            raise ValueError("Using sorted_indices with StepChunker is not supported.")
        elif initial_shuffle:
            raise ValueError("Using sorted_indices with initial_shuffle is not supported.")

        chunks = self.__chunker.chunk(self.__world_size, indices)
        subsets = [Subset(self._full_dataset, chunk) for chunk in chunks]
        assert len(subsets) == self.__world_size
        return subsets

    def __get_data_loaders(self, batch_size: Optional[int] = 1,
                           shuffle: Optional[bool] = None, sampler=None,
                           num_workers: int = 0, collate_fn=None,
                           pin_memory: bool = False, drop_last: bool = False,
                           timeout: float = 0, worker_init_fn=None,
                           multiprocessing_context=None, generator=None,
                           prefetch_factor: int = 2,
                           persistent_workers: bool = False,
                           pin_memory_device: str = ""):
        """Create a data loader for each rank."""
        data_loaders = []
        for subset in self.__subsets:
            data_loaders.append(DataLoader(subset, batch_size=batch_size,
                                           shuffle=shuffle, sampler=sampler,
                                           num_workers=num_workers, collate_fn=collate_fn,
                                           pin_memory=pin_memory, drop_last=drop_last,
                                           timeout=timeout, worker_init_fn=worker_init_fn,
                                           multiprocessing_context=multiprocessing_context,
                                           generator=generator,
                                           prefetch_factor=prefetch_factor,
                                           persistent_workers=persistent_workers,
                                           ))
        return data_loaders

    @property
    def data_loader(self):
        return self.__data_loaders[self.__rank] if not self.global_ else self._data_loader

    @property
    def global_(self):
        return self._global_

    def __create_and_store_histogram(self):
        """Create and store a histogram of the number of samples per label per rank"""
        gpu_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        label_tensors = [labels for _, labels in self.data_loader]
        labels = torch.cat(label_tensors, dim=0)
        self.__store_labels(labels)
        labels = labels.to(self.__rank % gpu_per_node)
        gather_list = [torch.empty_like(labels) for _ in range(self.__world_size)]
        dist.all_gather(gather_list, labels)
        if self.__rank == 0:
            rows = self.__world_size // gpu_per_node
            # Get dict of label -> count for each rank.
            list_of_label_to_count = \
                [get_label_to_count(labels.cpu().numpy()) for labels in gather_list]
            plot_helper = PlotHelper(list_of_label_to_count, gpu_per_node)
            # Create and save plots.
            plot_helper.histogram_raster(rows)
            plot_helper.per_rank_with_entropy()
            plot_helper.all_lines()

    def __store_labels(self, labels):
        """Store the labels in a csv using pandas"""
        folder_name = "labels-per-rank"
        if self.__rank == 0:
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
        df = pd.DataFrame(labels)
        dist.barrier()
        df.to_csv(f"{folder_name}/{self.__rank}.csv", index=False, header=False)

    def print0(self, message):
        if self.__rank == 0:
            print(message)

    def __repr__(self):
        """Print all attributes of the class."""
        return str(self.__dict__)
