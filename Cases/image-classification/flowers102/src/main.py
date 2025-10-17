import sys
# import os
from datetime import datetime
import logging

import yaml
from pydantic import ValidationError
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T

# Импорт из общего кода
from common import read_conf, find_file, save_npz, Trainer
from common import make_logger, visualize, make_parser
from common.models.cv_cls.alexnet import create_cnn
from common.models.cv_cls.resnet import make_resnet, LEVELS2FROZEN

# Локальный импорт
from utils.flower_data_manager import FlowerDataManager, ROOT
from utils.flower_dataset import FlowerDataSet


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

RESNET = True
LOAD_IMAGES = True
EPOCHS = 25
LR = 1e-4 # learning rate
CHECKPOINTS_DIR = ROOT / 'checkpoints'
# DEVICE = torch.device('cpu')
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if RESNET:
    # IM_SIZE = 232
    # BATCH_SIZE = 64

    IM_SIZE = 345
    BATCH_SIZE = 32

    name = 'resnet'
    FROZEN = 5 # (0, 2, 5, 6, 7, 8)
else:
    # IM_SIZE = 128
    # BATCH_SIZE = 64
    IM_SIZE = 256
    BATCH_SIZE = 16
    name = f'cnn_sz{IM_SIZE}'


logger = make_logger(name=name, log_dir=ROOT / 'logs', level=logging.DEBUG)
logger.info(f"\nNew experiment: {LOAD_IMAGES=}, {EPOCHS=}, {DEVICE=}, {IM_SIZE=}, {LR=}, {BATCH_SIZE=}")
if RESNET:
    logger.info(f"{FROZEN=}, {LEVELS2FROZEN=}")
    assert FROZEN in LEVELS2FROZEN

# Загрузка первичных данных
data_loader = FlowerDataLoader()
logger.debug(f"FlowerDataLoader found data with {data_loader.num_classes} classes")
train_idx, train_labels, val_idx, val_labels, test_idx, test_labels = data_loader.split(
        test_size=0.2, val_size=0.16)

paths = data_loader.get_paths()

augmentations = [
    T.RandomHorizontalFlip(0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.RandomRotation(20),
]

if RESNET:
    im_net_mean = (0.485, 0.456, 0.406)
    im_net_std = (0.229, 0.224, 0.225)
    test_transform = T.Normalize(mean=im_net_mean, std=im_net_std)
    train_transform = T.Compose([
        *augmentations,
        test_transform
    ])
else:
    test_transform = None
    train_transform = T.Compose([
        *augmentations
    ])

trainset = FlowerDataSet(paths, train_idx, train_labels,
        im_size=IM_SIZE, load_images=LOAD_IMAGES,
        transform=train_transform, max_cut=0.15,
    )
logger.debug(f"{len(trainset)=}")
valset = FlowerDataSet(paths, val_idx, val_labels,
        im_size=IM_SIZE, load_images=LOAD_IMAGES,
        transform=test_transform,
    )
logger.debug(f"{len(valset)=}")
testset = FlowerDataSet(paths, test_idx, test_labels,
        im_size=IM_SIZE, load_images=LOAD_IMAGES,
        transform=test_transform,
    )
logger.debug(f"{len(testset)=}")

train_dataloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(valset, batch_size=128, shuffle=False)
test_dataloader = DataLoader(testset, batch_size=128, shuffle=False)

if RESNET:
    # Инициализация модели обученой на датасете imagenet
    model = make_resnet(logger=logger, n_classes=data_loader.num_classes, frozen=FROZEN)
    # weights = ResNet50_Weights.IMAGENET1K_V2
    # logger.debug(f"Load {weights}...")
    # model = resnet50(weights=weights)
    # model = prepare_resnet(model, out_dim=data_loader.num_classes, logger=logger, frozen=FROZEN)
    # logger.debug("Weights loaded")
else:
    model = create_cnn(logger=logger, out_dim=data_loader.num_classes, im_size=IM_SIZE, fc_feats=512, conv_feats=None)

model = model.to(DEVICE)

if not RESNET:
    logger.debug(f"{model.FC2[1].weight.device=}")
    logger.debug(f"{model.FC2[1].weight.dtype=}")

# Обучение
trainer = Trainer(device=DEVICE, checkpoints_dir=CHECKPOINTS_DIR, logger=logger)

history = trainer.train_loop(model, train_dataloader, val_dataloader,
        lr=LR, epochs=EPOCHS,
        optimizer_type=torch.optim.Adamax,
        loss_type=nn.CrossEntropyLoss,
    )

visualize.compare_on_plot(history['train_metrics'], history['val_metrics'], logger=logger,
        name='accuracy', save_path=ROOT / 'plots' / 'last_history.png')

out_dict = {}
metric_on_test = trainer.evaluation(model, test_dataloader, out_dict=out_dict)
logger.info(f"On test data: accurary = {metric_on_test:.3f}")

# visualize.show_evaluation_results(
        # all_labels=out_dict['g_truth'],
        # all_preds=out_dict['predictions'],
        # classes=data_loader.flower_classes,
    # )
