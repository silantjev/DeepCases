import sys
import argparse
import os
from datetime import datetime
import logging

import yaml
from pydantic import ValidationError
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader

# Импорт из общего кода
from common.utils import find_file
from common.trainer import Trainer
from common.log import make_logger
from common import visualize
from common.models.nlp_cls.rnn import RNNModel
from common.models.nlp_cls.transformer import TransformerClassifier

# Локальный импорт
from utils.json_conf import read_conf
from utils.load_data import ROOT, save_npz
from utils.seq_dataset import make_dataloader, packed_collate_fn
from utils.config import Config

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
torch.multiprocessing.set_sharing_strategy('file_system')


def take_args():
    parser = argparse.ArgumentParser(description=f'Обучение модели', add_help=False) 

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='показать справку и выйти')
    parser.add_argument('--params', type=str, default=None, help='конфигурационный yaml-файл')
    parser.add_argument('--conf', '--split-conf', type=str, default=None, help='конфигурационный json-файл с процентами')

    return parser.parse_args()

args = take_args()
val_percent, test_percent = read_conf(path=args.conf)
train_percent = 100 - val_percent - test_percent
train_name = f'train{train_percent}'
val_name = f'val{val_percent}'
yaml_path = args.params
if yaml_path is None:
    yaml_path = "train_conf.yaml"
yaml_path = find_file(yaml_path, root=ROOT)

with open(yaml_path, "r", encoding='utf-8') as f:
    config_dict = yaml.safe_load(f)

try:
    config = Config(**config_dict)
except ValidationError as exc:
    print("[Error]: failed to parse yaml-file.")
    print("Validation errors:\n")
    print(exc.json(indent=2))
    print(f"\nCheck your configuration yaml-file \"{yaml_path}\"")
    sys.exit(1)

train_params = config.train_params

batch_size = config.train_params.batch_size
val_batch_size = config.train_params.val_batch_size

model_params = config.model_params[config.model]

n_layers = model_params.layers
n_heads = model_params.heads

# Дополнительные проверки
if n_heads != 0:
    if config.model == 'rnn' and model_params.bidirectional:
        raise ValueError(f"rnn with attention does not support case \"bidirectional: true\", change to false")
elif config.model != 'rnn':
    raise ValueError(f"transformer should have at least one head")

# MAX_LENGTH = 2113
FINAL_ATTENTION = True # for transformer

def f1_macro(gt, pred):
    return f1_score(gt, pred, average="macro")

metric = f1_macro
metric_name = metric.__name__ if hasattr(metric, "__name__") else 'metric'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PIN_MEMORY = False
# PIN_MEMORY = (str(DEVICE) == 'cpu')
EXPS_DIR = ROOT / 'exps'
if not EXPS_DIR.is_dir():
    EXPS_DIR.mkdir()

if config.model == 'rnn':
    name = model_params.net_class + f"_d{model_params.hidden_dim}"
else:
    name = f"trans_d{model_params.feedforward_dim}"

name += f"_l{model_params.layers}"

if n_heads:
    name += f"_att{n_heads}"
elif model_params.bidirectional:
    name += f"_D2"

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
params['data'] = {'train': train_name, 'val': val_name}
params['model'] = config.model
params['model_params'] = model_params.model_dump()
params['train_params'] = train_params.model_dump()
params['metric'] = metric_name

with open(exp_dir / 'params.yaml', 'w', encoding='utf-8') as f:
    yaml.safe_dump(
            params,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            indent=2,
        )

# Логирование
logger = make_logger(name=name, log_dir=ROOT / 'logs', level=logging.DEBUG)
logger.info("\n")
logger.info("New experiment. Model used: '%s'.\n\tModel parameters: %s;"
        "\n\tTrain parameters: %s; \n\tmetric name: %s.",
        config.model, repr(model_params),
        repr(config.train_params), metric_name)

# Загрузчики данных
trainloader = make_dataloader(
        path=train_name,
        batch_size=batch_size,
        pin_memory=PIN_MEMORY,
        for_train=True,
        )
trainset = trainloader.dataset
class_weights = trainset.class_weights
embedding_dim = trainset.embedding_dim
max_seq_len = trainset.max_length

logger.debug(f"Data with %d classes loaded", len(trainset.class_weights))
logger.info(f"{embedding_dim=}")
logger.debug(f"{len(trainset)=}")

# Отдельный загрузчик с теми же данными, но для подсчёта метрики в конце эпохи:
trainvalloader = DataLoader(
        trainset,
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=packed_collate_fn,
        pin_memory=PIN_MEMORY,
    )

valloader = make_dataloader(
        path=val_name,
        vector_path=train_name + '_vectors.npy',
        batch_size=val_batch_size,
        pin_memory=PIN_MEMORY,
        for_train=False,
        )
valset = valloader.dataset
logger.debug(f"{len(valset)=}")

if config.model.startswith('trans'):
    model = TransformerClassifier(
            num_classes=len(class_weights),
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            num_layers=n_layers,
            dim_feedforward=model_params.feedforward_dim,
            max_seq_len=max_seq_len,
            dropout=model_params.dropout,
            dropout_lin=model_params.dropout_lin,
            add_cls=True,
            final_attention=FINAL_ATTENTION,
        )
else:
    model = RNNModel(
            embedding_dim,
            output_dim=len(class_weights),
            hidden_dim=model_params.hidden_dim,
            rnn_class=model_params.net_class,
            bidirectional=model_params.bidirectional and (n_heads == 0),
            dropout=model_params.dropout,
            dropout_lin=model_params.dropout_lin,
            n_layers=n_layers,
            two_dense=True,
            n_heads = n_heads,
        )

model = model.to(device=DEVICE)

trainer = Trainer(device=DEVICE, best_checkpoint=exp_dir / (name + '_best.pt'), logger=logger)

history = trainer.train_loop(model, trainloader, valloader, trainvalloader=trainvalloader,
        weight=class_weights,
        metric=metric,
        lr=config.train_params.lr, epochs=config.train_params.epochs,
        optimizer_type=torch.optim.AdamW,
        loss_type=nn.CrossEntropyLoss,
        metric_during_epoch=False,
    )

# Имя фала для графика и для сохранения history

npz_path = exp_dir / (name + "_history.npz")
save_npz(path=npz_path, dict_to_save=history)

# Построим график и сохраним в png-файл
plot_path = exp_dir / (name + "_plot.png")
visualize.compare_on_plot(history, logger=logger,
        name=name, metric_name=metric_name, save_path=plot_path)

