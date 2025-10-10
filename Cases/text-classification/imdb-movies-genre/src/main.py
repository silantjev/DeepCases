import sys
import os
from datetime import datetime
import logging

from sklearn.metrics import f1_score

import torch
from torch import nn
from torch.utils.data import DataLoader

#local imports
from utils.load_data import ROOT, save_npz
from utils.seq_dataset import make_dataloader, packed_collate_fn
PROJECT_ROOT = ROOT
while (PROJECT_ROOT.name not in ['', '.', 'home', 'DeepCases']):
    PROJECT_ROOT = PROJECT_ROOT.parent

assert PROJECT_ROOT.name == 'DeepCases'

sys.path.insert(0, str(PROJECT_ROOT / 'common'))
from trainer import Trainer
from log import make_logger
import visualize
from models.nlp_cls.rnn import RNNModel
from models.nlp_cls.transformer import TransformerClassifier

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
torch.multiprocessing.set_sharing_strategy('file_system')

HISTORY_DIR = ROOT / 'history'
PLOTS_DIR = ROOT / 'plots'
if not HISTORY_DIR.is_dir():
    HISTORY_DIR.mkdir()

if not PLOTS_DIR.is_dir():
    PLOTS_DIR.mkdir()

# MAX_LENGTH = 2113

EPOCHS = 25
BATCH_SIZE = 32
VAL_BATCH = 32
LR = 1e-4 # learning rate

# HIDDEN_DIM = 1200 # FFN dim
HIDDEN_DIM = 128
DROPOUT = 0.1
DROPOUT_LIN = 0.2
N_LAYERS = 2
NET_CLASS = 'GRU'
# NET_CLASS = 'transformer'
N_HEADS = 0
BIDIRECTIONAL = True # for N_HEADS = 0
FINAL_ATTENTION = True # for transformer

def f1_macro(gt, pred):
    return f1_score(gt, pred, average="macro")

metric = f1_macro
metric_name = metric.__name__ if hasattr(metric, "__name__") else 'metric'


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PIN_MEMORY = False
# PIN_MEMORY = (str(DEVICE) == 'cpu')

CHECKPOINTS_DIR = ROOT / 'checkpoints'
if not CHECKPOINTS_DIR.is_dir():
    CHECKPOINTS_DIR.mkdir()

name = NET_CLASS + f"_l{N_LAYERS}_h{HIDDEN_DIM}"
if N_HEADS:
    name += f"_att{N_HEADS}"
elif BIDIRECTIONAL:
    name += f"_D2"

logger = make_logger(name=name, log_dir=ROOT / 'logs', level=logging.DEBUG)
logger.info("\n")
fa_str = ""
if NET_CLASS.startswith('trans') and FINAL_ATTENTION:
    fa_str = "+1"
    name += "_fa"
logger.info(f"New experiment: {EPOCHS=}, {DEVICE}, {LR=}, {BATCH_SIZE=}, {HIDDEN_DIM=}, {DROPOUT=}, {DROPOUT_LIN=}, {N_LAYERS=}{fa_str}, {NET_CLASS=}, {N_HEADS=}, {metric_name=}")

# Загрузчики данных
trainloader = make_dataloader(
        path='train75',
        batch_size=BATCH_SIZE,
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
        batch_size=VAL_BATCH,
        shuffle=False,
        collate_fn=packed_collate_fn,
        pin_memory=PIN_MEMORY,
    )

valloader = make_dataloader(
        path='val25',
        vector_path='train75_vectors.npy',
        batch_size=VAL_BATCH,
        pin_memory=PIN_MEMORY,
        for_train=False,
        )
valset = valloader.dataset
logger.debug(f"{len(valset)=}")

if NET_CLASS.startswith('trans'):
    assert N_HEADS
    model = TransformerClassifier(
            num_classes=len(class_weights),
            embedding_dim=embedding_dim,
            n_heads=N_HEADS,
            num_layers=N_LAYERS,
            dim_feedforward=HIDDEN_DIM,
            max_seq_len=max_seq_len,
            dropout=DROPOUT,
            dropout_lin=DROPOUT_LIN,
            add_cls=True,
            final_attention=FINAL_ATTENTION,
        )
else:
    model = RNNModel(
            embedding_dim,
            output_dim=len(class_weights),
            hidden_dim=HIDDEN_DIM,
            rnn_class=NET_CLASS,
            bidirectional=BIDIRECTIONAL and (N_HEADS == 0),
            dropout=DROPOUT,
            dropout_lin=DROPOUT_LIN,
            n_layers=N_LAYERS,
            two_dense=True,
            n_heads = N_HEADS,
        )

model = model.to(device=DEVICE)

trainer = Trainer(device=DEVICE, checkpoints_dir=CHECKPOINTS_DIR, logger=logger)

history = trainer.train_loop(model, trainloader, valloader, trainvalloader=trainvalloader,
        weight=class_weights,
        metric=metric,
        lr=LR, epochs=EPOCHS,
        optimizer_type=torch.optim.AdamW,
        loss_type=nn.CrossEntropyLoss,
        metric_during_epoch=False,
    )

# Имя фала для графика и для сохранения history
filename = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + ".png"

npz_path = HISTORY_DIR / filename
save_npz(path=npz_path, dict_to_save=history)

# Построим график и сохраним в png-файл
path = PLOTS_DIR / filename
visualize.compare_on_plot(history, logger=logger,
        name=name, metric_name=metric_name, save_path=path)

