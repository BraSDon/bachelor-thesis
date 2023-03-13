import abc
import os

from torchvision.datasets import ImageFolder
import torch.distributed as dist
from torch.utils.data import Dataset, Subset


class Transparent(metaclass=abc.ABCMeta):

    recording = True

    @abc.abstractmethod
    def next_epoch(self):
        pass

    @abc.abstractmethod
    def _record_access(self, index, path):
        pass

    def start_recording(self):
        self.recording = True

    def stop_recording(self):
        self.recording = False


def create_and_get_output_file(rank):
    folder_path = "accessed_indices"
    if rank == 0 and not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    # Make sure folder is created before proceeding.
    dist.barrier()
    file_path = f"{folder_path}/{rank}.txt"
    with open(file_path, "w") as f:
        f.write(f"Rank {rank} is running and recording accessed files.\n")
    return file_path


class TransparentFolder(ImageFolder, Transparent):
    def __init__(self, root: str, transform=None) -> None:
        self.recording = True
        assert dist.is_initialized()
        rank = dist.get_rank()
        self.file_path = create_and_get_output_file(rank)
        super().__init__(root, transform=transform)

    def __getitem__(self, index: int):
        """Exact copy of ImageFolder.__getitem__ except for the last line which records the access."""
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.recording: self._record_access(index, path)
        return sample, target

    def next_epoch(self):
        with open(self.file_path, "a") as f:
            f.write("\n")

    def _record_access(self, index, path):
        # trim path to only include the file name
        path = path.split("/")[-1]
        with open(self.file_path, "a") as f:
            f.write(f"({index},{path});")

    def __len__(self):
        """
        Make sure that len(dataset) % world_size == 0 to ensure that dataset is not modified in DataDistributer.
        Otherwise, recording of accessed indices will not work.
        """
        length = super(TransparentFolder, self).__len__()
        world_size = dist.get_world_size()
        if length % world_size != 0:
            return length - (length % world_size)
        else:
            return length


class TransparentDataset(Dataset, Transparent):
    def __init__(self, dataset):
        self.recording = True
        assert dist.is_initialized()
        rank = dist.get_rank()
        self.dataset = self.__trim_dataset(dataset)
        self.file_path = create_and_get_output_file(rank)

    @staticmethod
    def __trim_dataset(dataset):
        """
        Make sure that len(dataset) % world_size == 0 to ensure that dataset is not modified in DataDistributer.
        Otherwise, recording of accessed indices will not work.
        """
        world_size = dist.get_world_size()
        # Note: making sure that the dataset is evenly divisible by world_size
        # to avoid dataset being overriden in data_distributer.py
        length = len(dataset)
        if length % world_size != 0:
            new_length = length - (length % world_size)
            dataset = Subset(dataset, range(new_length))
        return dataset

    def __getitem__(self, index):
        if self.recording: self._record_access(index)
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def __getitems__(self, indices):
        if self.recording:
            for index in indices:
                self._record_access(index)
        return [self.dataset[index] for index in indices]

    def _record_access(self, index, path=None):
        with open(self.file_path, "a") as f:
            f.write(f"{index};")

    def next_epoch(self):
        with open(self.file_path, "a") as f:
            f.write("\n")
