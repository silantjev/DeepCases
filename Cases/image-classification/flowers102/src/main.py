import sys
# import os
from datetime import datetime
import logging

from pydantic import ValidationError
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T

# Импорт из общего кода
from common import read_conf, find_file, save_npz, Trainer
from common import make_logger, visualize
from common import make_arg_parser, read_yaml, save_yaml, cv_cls_config, print_pydantic_validation_errors
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

config_dict = read_yaml(yaml_path)
if config_dict is None:
    sys.exit(1)

try:
    config = cv_cls_config.Config(**config_dict)
except ValidationError as exc:
    print_pydantic_validation_errors(exc, yaml_path)
    sys.exit(1)

train_params = config.train_params

batch_size = config.train_params.batch_size
val_batch_size = train_params.val_batch_size

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
# print(f"{exp_dir=}")
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

# Логирование
logger = make_logger(name=name, log_dir=ROOT / 'logs', level=logging.DEBUG)
logger.info("\n")
logger.info("New experiment. Model used: '%s'.\n\tModel parameters: %s;"
        "\n\tTrain parameters: %s; \n\tmetric name: %s.",
        config.model, repr(model_params),
        repr(config.train_params), metric_name)

# Загрузка первичных данных
data_loader = FlowerDataManager()
logger.debug("FlowerDataManager found data with %d classes", data_loader.num_classes)
train_idx, train_labels, val_idx, val_labels, test_idx, test_labels = data_loader.split(
        test_size=test_percent / 100,
        val_size=val_percent / 100,
    )

image_paths = data_loader.get_paths()

# Функторы предобработки и аугментации
augmentations = [
    T.RandomHorizontalFlip(0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.RandomRotation(20),
]

if config.model == 'alexnet':
    train_transform = T.Compose([
        *augmentations
    ])
    test_transform = None
else: # resnet
    logger.info("frozen = %d; possible values are %s", model_params.frozen, str(set(LEVELS2FROZEN)))
    im_net_mean = (0.485, 0.456, 0.406)
    im_net_std = (0.229, 0.224, 0.225)
    imagenet_norm = T.Normalize(mean=im_net_mean, std=im_net_std)
    train_transform = T.Compose([
        *augmentations,
        imagenet_norm
    ])
    test_transform = imagenet_norm

# Загрузчики данных:
trainset = FlowerDataSet(image_paths, train_idx, train_labels,
        im_size=model_params.image_size,
        load_images=train_params.load_images,
        transform=train_transform, max_cut=0.15,
    )
logger.debug(f"Train dataset with size %d created", len(trainset))
valset = FlowerDataSet(image_paths, val_idx, val_labels,
        im_size=model_params.image_size,
        load_images=train_params.load_images,
        transform=test_transform,
    )
logger.debug(f"Validation dataset with size %d created", len(valset))
testset = FlowerDataSet(image_paths, test_idx, test_labels,
        im_size=model_params.image_size,
        load_images=train_params.load_images,
        transform=test_transform,
    )
logger.debug(f"Test dataset with size %d created", len(testset))

train_dataloader = DataLoader(trainset, batch_size=train_params.batch_size, shuffle=True)
val_dataloader = DataLoader(valset, batch_size=train_params.val_batch_size, shuffle=False)
test_dataloader = DataLoader(testset, batch_size=train_params.val_batch_size, shuffle=False)

if config.model == 'alexnet':
    model = create_cnn(
            logger=logger,
            out_dim=data_loader.num_classes,
            im_size=model_params.image_size,
            fc_feats=model_params.fc_feat,
            conv_feats=model_params.conv_feats,
            dropout=model_params.dropout,
        )
else: # Инициализация модели resnet обученой на датасете imagenet
    model = make_resnet(
            logger=logger,
            n_classes=data_loader.num_classes,
            frozen=model_params.frozen,
        )

model = model.to(DEVICE)

# Обучение
logger.info('Experiment directory: \"%s\"', exp_dir)
logger.info('Model: %s', model)

trainer = Trainer(device=DEVICE, best_checkpoint=exp_dir / (name + '_best.pt'), logger=logger)

history = trainer.train_loop(model, train_dataloader, val_dataloader,
        metric=metric,
        lr=config.train_params.lr,
        epochs=config.train_params.epochs,
        optimizer_type=torch.optim.Adamax,
        loss_type=nn.CrossEntropyLoss,
    )

# Имя фала для графика и для сохранения history
npz_path = exp_dir / (name + "_history.npz")
save_npz(path=npz_path, dict_to_save=history)

# Построим график и сохраним в png-файл
plot_path = exp_dir / (name + "_plot.png")
visualize.compare_on_plot(history, logger=logger,
        name=name, metric_name=metric_name, save_path=plot_path)

out_dict = {}
metric_on_test = trainer.evaluation(model, test_dataloader, out_dict=out_dict)
logger.info("On test data: %s = %.3f", metric_name, metric_on_test)

# visualize.show_evaluation_results(
        # all_labels=out_dict['g_truth'],
        # all_preds=out_dict['predictions'],
        # classes=data_loader.flower_classes,
    # )
