import sys
import logging
from pathlib import Path
import torch

from utils.imdb_data_manager import ROOT
from utils.seq_dataset import make_dataloader
from common import trainer, make_arg_parser, read_conf, read_yaml, find_file, make_logger
from common import f1_macro, accuracy
from common.models.nlp_cls.transformer import TransformerClassifier
from common.models.nlp_cls.rnn import RNNModel

MAX_LENGTH = 2113

def find_pt_and_yaml(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"File or directory \"{model_path}\" not found")

    if model_path.is_dir():
        dir_path = model_path
        templ = '*_best.pt'
        paths = list(dir_path.glob(templ))
        if len(paths) == 0:
            raise FileNotFoundError(f"File \"{dir_path}/{templ}\" not found")
        if len(paths) > 1:
            raise ValueError(f"Directory \"{dir_path}\" contains more than 1 file \"{templ}\". Write full name of the pt-file")
        model_path = paths[0]
    else:
        dir_path = model_path.parent

    yaml_path = dir_path / 'params.yaml'
    if not yaml_path.is_file():
        raise FileNotFoundError(f"File \"{model_path}\" not found")

    return model_path, yaml_path


parser = make_arg_parser(description='Оценка модели', params=False)
parser.add_argument('model', type=str, help='Файл с весами модели, рядом с которой лежит файл "params.yaml", или папка с этими файлами')
args = parser.parse_args()

model_path = find_file(args.model, root=ROOT, dir_allowed=True)
model_path, yaml_path = find_pt_and_yaml(model_path)

params = read_yaml(yaml_path)
if params is None:
    sys.exit(1)


max_seq_len = params.get('max_seq_len', MAX_LENGTH)
n_classes = params.get('n_classes', 27)
model_name = params['model']

model_params = params['model_params']
n_heads = model_params['heads']
train_params = params['train_params']
val_batch_size = train_params['val_batch_size']

# Логирование
logger = make_logger(name=params['name'], log_dir=ROOT / 'logs', level=logging.DEBUG)
logger.info("\n")
logger.info("Validation of the model '%s'", model_path.name)

# Загрузчики данных
val_percent, test_percent = read_conf(root=ROOT, path=args.conf)
train_percent = 100 - val_percent - test_percent
train_name = f'train{train_percent}'
val_name = f'val{val_percent}'
test_name = f'test{test_percent}'

valloader = make_dataloader(
        path=val_name,
        vector_path=train_name + '_vectors.npy',
        batch_size=val_batch_size,
        pin_memory=False,
        for_train=False,
        )

logger.info("Validation data loaded. Dataset size is %d", len(valloader.dataset))

embedding_dim = valloader.dataset.embedding_dim

if model_name == 'rnn':
    model = RNNModel(
            embedding_dim,
            output_dim=n_classes,
            hidden_dim=model_params['hidden_dim'],
            rnn_class=model_params['net_class'],
            bidirectional=model_params['bidirectional'] and (n_heads == 0),
            dropout=0,
            dropout_lin=0,
            n_layers=model_params['layers'],
            two_dense=True,
            n_heads = n_heads,
        )
else:
    model = TransformerClassifier(
                num_classes=n_classes,
                embedding_dim=embedding_dim,
                n_heads=n_heads,
                num_layers=model_params['layers'],
                dim_feedforward=model_params['feedforward_dim'],
                max_seq_len=max_seq_len,
                dropout=0,
                dropout_lin=0,
                add_cls=True,
                final_attention=True,
            )

model.load_state_dict(torch.load(model_path, map_location='cpu'))
logger.info("Model weights loaded from '%s'", model_path)
trainer.freeze(model)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(DEVICE)

trainer = trainer.Trainer(device='cpu', best_checkpoint=None, logger=logger)


out_dict = {}
metrics_on_val = trainer.evaluation(
    model,
    valloader,
    out_dict=out_dict,
    metrics=[f1_macro, accuracy],
)

logger.info("On validation data: F1 (macro) = %.3f", metrics_on_val[0])
logger.info("On validation data: accuracy = %.3f", metrics_on_val[1])

if test_percent:
    testloader = make_dataloader(
            path=test_name,
            vector_path=train_name + '_vectors.npy',
            batch_size=val_batch_size,
            pin_memory=False,
            for_train=False,
            )

    logger.info("Test data loaded. Dataset size is %d", len(testloader.dataset))

    metrics_on_test = trainer.evaluation(
        model,
        testloader,
        out_dict=out_dict,
        metrics=[f1_macro, accuracy],
    )

    logger.info("On validation data: F1 (macro) = %.3f", metrics_on_test[0])
    logger.info("On validation data: accuracy = %.3f", metrics_on_test[1])
else:
    logger.info("No test data, since test_percent = 0")


