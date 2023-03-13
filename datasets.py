import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torchvision.datasets as datasets
import torch.distributed as dist

from util.transparent import TransparentFolder, TransparentDataset
from util.chunker import SequentialChunker, StepChunker
from util.data_distributer import DataDistributer
from util.dataset_info import DATASET_INFO

# Bug fix: https://discuss.pytorch.org/t/oserror-image-file-is-truncated-150-bytes-not-processed/64445
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataHandler:
    def __init__(self, dataset_name, batch_size=1, workers=1, shuffle=False,
                 initial_shuffle=False, chunker_name=None, verbose=True, global_shuffle=False, stage=True, transparent=False):
        self.dataset_path = self.__get_path(dataset_name)
        self.chunker = self.__get_chunker(chunker_name)

        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.workers = workers
        self.shuffle = shuffle
        self.initial_shuffle = initial_shuffle
        self.chunker_name = chunker_name
        self.verbose = verbose
        self.global_shuffle = global_shuffle
        self.transparent = transparent

        if stage:
            self.dataset_path = self.copy_dataset_to_tmp()

        self.train_dataset, self.val_dataset = self.__get_datasets()
        self.train_distributer, self.val_distributer = self.__get_distributers()

    def __get_datasets(self):
        train_path = self.dataset_path + "train"
        val_path = self.dataset_path + "val"
        train_transform = DATASET_INFO[self.dataset_name]["train_transform"]
        val_transform = DATASET_INFO[self.dataset_name]["val_transform"]

        # NOTE: Hardcoded for CIFAR10. Should be generalized if possible.
        if self.dataset_name == "cifar10":
            train_dataset = datasets.CIFAR10(root=self.dataset_path, train=True, transform=train_transform)
            if self.transparent:
                train_dataset = TransparentDataset(train_dataset)
            val_dataset = datasets.CIFAR10(root=self.dataset_path, train=False, transform=val_transform)
            return train_dataset, val_dataset

        # Only track the indices of the training dataset if transparent
        # WARNING: Using TransparentFolder only works for ImageFolder datasets!!!
        if self.transparent:
            train_dataset = TransparentFolder(train_path, transform=train_transform)
        else:
            train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_path, transform=val_transform)
        return train_dataset, val_dataset

    def __get_distributers(self):
        train_distributer = DataDistributer(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                            initial_shuffle=self.initial_shuffle, chunker=self.chunker,
                                            verbose=self.verbose, global_=self.global_shuffle, num_workers=self.workers)
        val_distributer = DataDistributer(self.val_dataset, batch_size=self.batch_size,
                                          verbose=False, num_workers=self.workers)
        return train_distributer, val_distributer

    @staticmethod
    def __get_chunker(chunker_name):
        if chunker_name == "seq":
            chunker = SequentialChunker()
        elif chunker_name == "step":
            chunker = StepChunker()
        else:
            raise ValueError(f"Chunker {chunker_name} not supported.")
        return chunker

    @staticmethod
    def __get_path(dataset_name):
        try:
            return DATASET_INFO[dataset_name]["path"]
        except KeyError:
            raise ValueError(f"Dataset {dataset_name} not supported.")

    def copy_dataset_to_tmp(self):
        TMP = os.environ["TMP"]
        src_dir = self.dataset_path
        dst_dir = f"{TMP}/{self.dataset_name}"
        # Copy dataset only once per node
        global_rank = dist.get_rank()
        local_rank = global_rank % int(os.environ["SLURM_GPUS_ON_NODE"])
        if local_rank == 0:
            start = time.time()
            print(f"[GPU {global_rank}] Copying dataset {self.dataset_name} from {src_dir} to {dst_dir}...")
            self.copy_files(src_dir, dst_dir)
            print(f"[GPU {global_rank}] Finished copying files in {time.time() - start} seconds.")
        dist.barrier()
        return dst_dir
    
    @staticmethod
    def copy_file(src_file, dst_file):
        shutil.copy2(src_file, dst_file)

    def copy_files(self, src_dir, dst_dir, max_workers=32):
        # Create the destination directory if it doesn't exist
        os.makedirs(dst_dir, exist_ok=True)

        # Get a list of all files and directories in the source directory
        for dirpath, dirnames, filenames in os.walk(src_dir):
            # Create the corresponding subdirectories in the destination directory
            subdir = dirpath.replace(src_dir, dst_dir)
            os.makedirs(subdir, exist_ok=True)

            # Create a ThreadPoolExecutor with a max of 4 worker threads
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit the copy_file function for each file in the list
                futures = [executor.submit(self.copy_file, os.path.join(dirpath, f), os.path.join(subdir, f)) for f in
                           filenames]
                for future in as_completed(futures):
                    future.result()
