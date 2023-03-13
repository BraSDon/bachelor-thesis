import shutil
from time import time

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import pandas as pd
import os

from util.data_distributer import DataDistributer


class Runner:
    """
    Runs training and validation and stores the results.
    """
    def __init__(self, model: torch.nn.Module, train_distributer: DataDistributer,
                 val_distributer: DataDistributer, criterion, optimizer: torch.optim.Optimizer,
                 lr_scheduler, verbose=True):
        self.gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        self.global_rank = int(os.environ["SLURM_PROCID"])
        self.local_rank = self.global_rank % self.gpus_per_node

        self.model = model.to(self.local_rank)
        self.model = DDP(self.model, device_ids=[self.local_rank])

        self.train_distributer = train_distributer
        self.val_distributer = val_distributer

        # NOTE: if criterion is stateful, it needs to be moved to GPU
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.verbose = verbose

        # holds train_results for every epoch
        self.train_results = pd.DataFrame()

        # holds val_results for every epoch
        self.val_results = pd.DataFrame()

    def train(self, max_epochs: int, print_freq, store_model=False):
        if self.global_rank == 0:
            print("Starting training...")
            print(f"[GPU0] Batchsize: {self.train_distributer.data_loader.batch_size} | "
                  f"Steps: {len(self.train_distributer.data_loader)}")
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            self._validate()
            if epoch % print_freq == 0 and self.global_rank == 0:
                print("Training results so far... \n")
                print(self.train_results)

            if self.lr_scheduler is not None:
                # Call learning rate scheduler on val_loss metric
                self.lr_scheduler.step(self.val_results["val_loss"].iloc[-1], epoch=epoch)

        if store_model:
            self._save_checkpoint()

    def _run_epoch(self, epoch):
        start = time()
        self.model.train()

        # Set epoch in DistributedSampler to ensure new shuffling every epoch.
        if self.train_distributer.global_:
            self.train_distributer.data_loader.sampler.set_epoch(epoch)

        correct = 0
        total = 0
        loss_sum = 0
        for inputs, targets in self.train_distributer.data_loader:
            inputs = inputs.to(self.local_rank)
            targets = targets.to(self.local_rank)
            outputs, _ = self._run_batch(inputs, targets)

            correct, loss_sum, total = \
                self._update_epoch_statistics(correct, targets, loss_sum, outputs, total)

        end = time()

        self._calc_and_store_stats(loss_sum, correct, total,
                                   in_training=True, epoch_time=end - start)

        # If verbose, we need to indicate the end of the epoch in the log.
        if not self.verbose: return
        try:
            self.train_distributer._full_dataset.next_epoch()
        except AttributeError:
           pass

    def _run_batch(self, inputs, targets):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return outputs, loss.item()

    def _validate(self):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            loss_sum = 0
            for inputs, targets in self.val_distributer.data_loader:
                inputs = inputs.to(self.local_rank)
                targets = targets.to(self.local_rank)
                outputs = self.model(inputs)

                correct, loss_sum, total = self._update_epoch_statistics(correct, targets, loss_sum, outputs, total)
                del inputs, targets, outputs

        self._calc_and_store_stats(loss_sum, correct, total, in_training=False)

    def _update_epoch_statistics(self, correct, targets, loss_sum, outputs, total):
        loss_sum += self.criterion(outputs, targets).item()
        total += targets.size(0)
        correct += self._get_num_correct_predictions(targets, outputs)
        return correct, loss_sum, total

    def _calc_and_store_stats(self, loss_sum, correct_predictions, samples_processes, in_training,
                              epoch_time=0.0):
        """Calculate and store average (over ranks) loss and accuracy for epoch."""
        df, steps_in_epoch = self._get_dataframe_and_steps(in_training)
        loss = self._calc_avg_epoch_loss(loss_sum, steps_in_epoch)
        acc = self._calc_accuracy(correct_predictions, samples_processes)

        # Move to CPU, because pandas requires it.
        loss = loss.cpu()
        acc = acc.cpu()
        acc = acc.item()

        if in_training:
            assert epoch_time != 0
            temp_df = pd.DataFrame({"loss": loss, "accuracy": acc, "time": epoch_time}, index=[0])
            self.train_results = pd.concat([df, temp_df], ignore_index=True)
        else:
            temp_df = pd.DataFrame({"val_loss": loss, "val_accuracy": acc}, index=[0])
            self.val_results = pd.concat([df, temp_df], ignore_index=True)

    def _get_dataframe_and_steps(self, in_training):
        if in_training:
            df = self.train_results
            steps_in_epoch = len(self.train_distributer.data_loader)
        else:
            df = self.val_results
            steps_in_epoch = len(self.val_distributer.data_loader)
        return df, steps_in_epoch

    def _calc_avg_epoch_loss(self, loss_sum, steps_in_epoch):
        """
        Calculate average loss of epoch over all ranks.
        """
        avg_loss_on_node = loss_sum / steps_in_epoch
        avg_loss_on_node = torch.FloatTensor([avg_loss_on_node]).to(self.local_rank)
        return self._calculate_avg_over_world(avg_loss_on_node)

    def _calc_accuracy(self, correct, total):
        """
        Calculate total accuracy over all ranks in percent.
        """
        training_summary = torch.FloatTensor([correct, total]).to(self.local_rank)
        dist.all_reduce(training_summary, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
        correct, total = training_summary[0], training_summary[1]
        return 100 * correct / total

    def _save_checkpoint(self, is_best=False, filename="checkpoint.pth.tar"):
        """
        Checkpoint current model, old checkpoint will be overwritten. If is_best, store separately.
        """
        if self.global_rank != 0: return
        torch.save(self.model.module.state_dict(), filename)
        print(f"Stored model at {filename}")
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')

    @staticmethod
    def _get_num_correct_predictions(targets, outputs):
        _, predicted = torch.max(outputs.data, 1)
        return (predicted == targets).sum().item()

    @staticmethod
    def _calculate_avg_over_world(gpu_tensor):
        dist.all_reduce(gpu_tensor, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
        return gpu_tensor / dist.get_world_size()
