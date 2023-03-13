import os
import random

import numpy as np
import pandas as pd

import torch
import torch.distributed as dist
import torchvision.models.resnet
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import datasets
from util.warmup_lr_scheduler import WarmupLRScheduler
from runner import Runner
from util.parser import NNParser


def main():
    args = NNParser().parse_args()

    set_seed(args)
    ddp_setup()
    print0(args)
    if args.sanity_check: sanity_check()
    model, criterion, optimizer, lr_scheduler = get_training_objects(args)
    train_distributer, val_distributer = get_dataloading_objects(args)
    runner = Runner(model, train_distributer, val_distributer, criterion, optimizer, lr_scheduler, verbose=args.verbose)

    if args.init_only: return
    runner.train(args.epochs, args.print_freq, store_model=False)
    print_and_store_output(args, runner.train_results, runner.val_results)


def get_training_objects(args):
    """Create model, criterion, optimizer, and lr_scheduler."""
    model = create_model(args)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)

    if args.lr_scheduler is None:
        lr_scheduler = None
    else:
        if args.lr_scheduler != "ReduceLROnPlateau":
            raise NotImplementedError("Only ReduceLROnPlateau implemented")
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        # Use wrapper for warmup phase.
        if args.warmup_epochs > 0:
            if args.reference_kn == 0: raise ValueError("reference kn can not be 0")
            # lr after warmup phase. Using strategy from https://arxiv.org/abs/1706.02677.
            target_lr = args.lr * args.batch_size * dist.get_world_size() / args.reference_kn
            lr_scheduler = WarmupLRScheduler(lr_scheduler, target_lr=target_lr, start_lr=args.lr, verbose=args.verbose)

    return model, criterion, optimizer, lr_scheduler


def create_model(args):
    if args.arch == "alexnet":
        return torchvision.models.alexnet()
    elif args.arch == "resnet50":
        return torchvision.models.resnet.resnet50()
    else:
        NotImplementedError("Unknown model architecture")


def get_dataloading_objects(args):
    """Create train and val data distributers."""
    data_handler = datasets.DataHandler(args.dataset_name, batch_size=args.batch_size, workers=args.workers, shuffle=args.shuffle,
                                        initial_shuffle=args.initial_shuffle, chunker_name=args.chunker, verbose=args.verbose,
                                        global_shuffle=args.global_shuffle, stage=args.stage, transparent=args.transparent)
    return data_handler.train_distributer, data_handler.val_distributer


def set_seed(args):
    """
    Set seed for reproducibility. 
    Not sure if this suffices for reproducibility because of PyTorch internals.
    """
    seed = args.seed
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ddp_setup():
    """Setup distributed training."""
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NPROCS"])

    addr = os.environ["SLURM_LAUNCH_NODE_IPADDR"]
    port = "29500"
    os.environ["MASTER_PORT"] = port
    os.environ["MASTER_ADDR"] = addr

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def print_and_store_output(args, train_df, val_df):
    """Print and store training and validation dataframe (containing acc and loss)."""
    name = args.output_name + str(random.randint(0, 10000))
    out_name = f"{name}.csv"
    dist.barrier()
    if dist.get_rank() == 0:
        df = pd.concat([train_df, val_df], axis=1)

        print("============== output info =================================")
        print(df)
        print("===========================================================\n\n")

        args.world_size = dist.get_world_size()
        storedata = pd.HDFStore(out_name)
        storedata.put('data', df)
        storedata.get_storer('data').attrs.metadata = args
        storedata.close()


def sanity_check():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(
        f"Hello from rank {rank} of {world_size} where there are" f" {gpus_per_node} allocated GPUs per node.")

    local_rank = rank % gpus_per_node
    print(f"My local rank is: {local_rank}")
    print(f"Cuda available: {torch.cuda.is_available()}")

    disttest = torch.ones(1).to(local_rank)
    dist.all_reduce(disttest)
    disttest = int(disttest.item())
    print("test:", disttest)
    assert disttest == world_size
    dist.barrier()
    if rank == 0:
        print("--------------SANITY CHECK COMPLETE--------------")


def print0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)


if __name__ == '__main__':
    main()
