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
from common import make_logger, visualize
from common import make_arg_parser, read_yaml, save_yaml, cv_cls_config
from common.models.cv_cls.alexnet import create_cnn
from common.models.cv_cls.resnet import make_resnet, LEVELS2FROZEN
from common import accuracy

# Локальный импорт
from utils.flower_data_manager import FlowerDataManager, ROOT
from utils.flower_dataset import FlowerDataSet


DEFAULT_YAML_FILENAME = "train_conf.yaml"

args = make_arg_parser(description='Обучение модели', params=True).parse_args()
val_percent, test_percent = read_conf(root=ROOT, path=args.conf)
train_percent = 100 - val_percent - test_percent
yaml_path = args.params
if yaml_path is None:
    yaml_path = DEFAULT_YAML_FILENAME
yaml_path = find_file(yaml_path, root=ROOT)

config = read_yaml(yaml_path=yaml_path, Config=cv_cls_config.Config)
if config is None:
    sys.exit(1)

train_params = config.train_params

batch_size = config.train_params.batch_size
val_batch_size = config.train_params.val_batch_size

model_params = config.model_params[config.model]

metric = accuracy
metric_name = metric.__name__ if hasattr(metric, "__name__") else 'metric'

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

EXPS_DIR = ROOT / 'exps'
if not EXPS_DIR.is_dir():
    EXPS_DIR.mkdir()

name = config.model
name += f"_sz{model_params.image_size}"

if config.model == 'alexnet':
    name += f"_fc{model_params.fc_feat}"
else:
    name += f"_fr{model_params.frozen}"

exp_name = name + datetime.now().strftime("_%Y-%m-%d_%H:%M:%S")
exp_dir = EXPS_DIR / exp_name
print(f"{exp_dir=}")
assert not exp_dir.exists(), f"Folder \"{exp_dir}\" already exists"
exp_dir.mkdir()

params = dict()
params['params_file'] = yaml_path.name
params['name'] = name
params['date'] = datetime.now().strftime("%Y-%m-%d")
params['time'] = datetime.now().strftime("%H:%M")
params['data'] = {'train_percent': train_percent, 'val_percent': val_percent}
params['model'] = config.model
params['model_params'] = model_params.model_dump()
params['train_params'] = train_params.model_dump()
params['metric'] = metric_name

exp_yaml_path = exp_dir / 'params.yaml'
ok = save_yaml(exp_yaml_path, params)
if ok:
    print(f"Parameters saved to \"{exp_yaml_path}\"")
else:
    print(f"Failed to save parameters to \"{exp_yaml_path}\"")

logger = make_logger(name=name, log_dir=ROOT / 'logs', level=logging.DEBUG)
logger.info(f"\nNew experiment: {LOAD_IMAGES=}, {EPOCHS=}, {DEVICE=}, {IM_SIZE=}, {LR=}, {BATCH_SIZE=}")
if RESNET:
    logger.info(f"{FROZEN=}, {LEVELS2FROZEN=}")
    assert FROZEN in LEVELS2FROZEN

# Загрузка первичных данных
data_loader = FlowerDataManager()
logger.debug(f"FlowerDataManager found data with {data_loader.num_classes} classes")
train_idx, train_labels, val_idx, val_labels, test_idx, test_labels = data_loader.split(
        test_size=test_percent / 100,
        val_size=val_percent / 100,
    )

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
