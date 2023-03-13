from torchvision.transforms import transforms

# NOTE: Add new datasets here
###############################################################################
IMAGENET_PATH = "/pfs/work7/workspace/scratch/tz6121-shuffling/datasets/CLS-LOC/"
IMAGENET_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
IMAGENET_TRAIN_TRANSFORM = transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            IMAGENET_NORMALIZE,
        ])
IMAGENET_VAL_TRANSFORM = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            IMAGENET_NORMALIZE,
        ])
###############################################################################
CIFAR10_PATH = "/pfs/work7/workspace/scratch/tz6121-shuffling/datasets/cifar10"
CIFAR10_NORMALIZE = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
CIFAR10_TRAIN_TRANSFORM = CIFAR10_VAL_TRANSFORM = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        CIFAR10_NORMALIZE
    ])
###############################################################################

DATASET_INFO = {"imagenet":
                    {"path": IMAGENET_PATH,
                     "train_transform": IMAGENET_TRAIN_TRANSFORM,
                     "val_transform": IMAGENET_VAL_TRANSFORM},
                "cifar10":
                    {"path": CIFAR10_PATH,
                     "train_transform": CIFAR10_TRAIN_TRANSFORM,
                     "val_transform": CIFAR10_VAL_TRANSFORM}
                }