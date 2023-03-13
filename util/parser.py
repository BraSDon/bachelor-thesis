import argparse


class NNParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='PyTorch training')
        self._add_arguments(self.parser)

    def parse_args(self):
        return self.parser.parse_args()

    @staticmethod
    def _add_arguments(parser):
        # dataset and model
        parser.add_argument('--dataset-name', default="imagenet")
        parser.add_argument('--arch', default='resnet50',
                            help='model architecture (default: resnet50)')

        # data loading
        parser.add_argument('--shuffle', action="store_true", default=False)
        parser.add_argument("--initial-shuffle", action="store_true", default=False)
        parser.add_argument("--chunker", choices=["seq", "step"], default="seq")
        parser.add_argument("--use-sorted-dataset", action="store_true", default=False,
                            help="use the sorted indices of specified dataset. "
                                 "Caution: this option requires manual configuration in datasets.py")
        parser.add_argument('--workers', default=2, type=int,
                            help='number of data loading workers (default: 2)')
        parser.add_argument('--global-shuffle', action="store_true", default=False)
        parser.add_argument('--stage', action="store_true", default=False)
        parser.add_argument('--transparent', action="store_true", default=False)

        # hyper-parameters
        parser.add_argument('--epochs', default=40, type=int, help='number of total epochs to run')
        parser.add_argument('--batch-size', default=256, type=int,
                            help='mini-batch size (default: 256), each GPU processes this many '
                                 'samples in each iteration. Therefore we process N_GPUS * BATCH_SIZE samples per step')
        parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
        parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
        parser.add_argument('--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')

        # LR scheduler
        parser.add_argument('--warmup-epochs', default=5, type=int, help='number of warmup epochs (default: 5).'
                                                                         'no effect if lr_scheduler is not specified.')
        parser.add_argument("--lr-scheduler", type=str, default=None,
                            help="learning rate scheduler (avail: ReduceLROnPlateau, default: None)")

        # other
        parser.add_argument('--seed', default=None, type=int,help='seed for initializing training.')
        parser.add_argument('-p', '--print-freq', default=5, type=int,
                            help='epochs between printing training results (default: 5)')
        parser.add_argument("--reference-kn", type=int, default=256,
                            help="kn = num_workers * batch_size (default: 256). "
                                 "Based on paper https://arxiv.org/abs/1706.02677")
        parser.add_argument('--output-name', type=str, default="default",
                            help='name of the output file')

        # testing
        parser.add_argument('--init-only', action="store_true", default=False)
        parser.add_argument('--sanity-check', action="store_true", default=False)
        parser.add_argument("--verbose", action="store_true", default=False)
