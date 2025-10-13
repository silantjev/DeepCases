import numpy as np
from tqdm import tqdm
import torch
from torch import nn

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True

def accuracy(gt, pred):
    assert len(gt) == len(pred)
    return (gt == pred).mean()

class Trainer:
    """ Обучение и оценка моделей """
    def __init__(self, device, best_checkpoint, logger):
        self.device = device
        # if str(device).lower() != 'cpu':
            # torch.cuda.set_per_process_memory_fraction(0.5)
        self.best_checkpoint = best_checkpoint 
        self.logger = logger
        self.train_metric = 0
        self.len_trainset = 0

    def predict(self, x, model=None):
        """ Делаем предсказание класса через arg-max:
             - если model=None, то x — это уже вероятности (score)
             - иначе x — это входные данные модели, по которым model даёт вероятности
        """
        if model is not None:
            x = model(x.to(self.device)).cpu().detach()
        else: # x is already predicted probabilities
            x = x.cpu().detach()
        assert x.dim() == 2, f"probabilities/scores are supposed to be of shape [n_batch, n_classes], but we got {list(x.shape)}"
        return np.argmax(x.numpy(), axis=1)

    @torch.no_grad()
    def evaluation(self, model, dataloader, metric=None, metrics=None, out_dict=None):
        """ Оценка модели model по данным dataloader
            по одной метрике metric или нескольким metrics
            Результат предсказаний и метки сохраняются в out_dict и в self.last_predictions
            Возвращает полученное значение метрики или их список в том же порядке, что и в metrics
        """
        if metrics is None:
            metrics = []
        if not metric is None:
            assert len(metrics) == 0
            metrics = [metric]
        model.eval()
        if out_dict is None:
            out_dict = {}
        out_dict['predictions'] = []
        out_dict['g_truth'] = []
        for x, y in tqdm(dataloader):
            pred = self.predict(x.to(self.device), model)
            gt = y.cpu().detach().numpy()
            out_dict['predictions'].append(pred)
            out_dict['g_truth'].append(gt)
        for k in ['predictions', 'g_truth']:
            out_dict[k] = np.concatenate(out_dict[k])
        metric_values = []
        for metric in metrics:
            metric_value = metric(out_dict['g_truth'], out_dict['predictions'])
            if type(metric_value) not in [float, int]:
                metric_value = metric_value.item()
            metric_values.append(metric_value)

        if str(self.device).lower() != 'cpu':
            torch.cuda.empty_cache()
        self.last_predictions = dict(predictions=out_dict['predictions'], g_truth=out_dict['g_truth'])
        if len(metric_values) == 1:
            return metric_value
        return metric_values

    def train_epoch(self, model, optimizer, dataloader, loss, metric=None):
        """ Одна эпоха обучения """
        model.train()
        optimizer.zero_grad()
        if self.len_trainset:
            assert metric is not None
            self.train_metric = 0
            self.total_len = 0
        for x, y in tqdm(dataloader):
            scores = model(x.to(self.device))
            if scores.shape != y.shape + (scores.shape[1],):
                self.logger.error("Wrong shapes: scores.shape = %s, y.shape = %s", scores.shape, y.shape)
            loss_value = loss(scores, y.to(self.device))
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
            if self.len_trainset:
                pred = self.predict(scores.detach())
                gt = y.cpu().detach().numpy()
                self.train_metric += metric(gt, pred) * len(y)
                self.total_len += len(y)
        if str(self.device).lower() != 'cpu':
            torch.cuda.empty_cache()

    def train_loop(self,
        model, trainloader, valloader, trainvalloader=None, metric_during_epoch=False,
        metric=accuracy, weight=None,
        lr=0.0001, epochs=10, optimizer_type=torch.optim.Adam, loss_type=nn.CrossEntropyLoss,
    ):
        """
        Основная функция для обучения
        После каждой эпохи считаем метрику metric на train и на val
        Функция возвращает словарь с сохранёнными метриками
        """
        optimizer = optimizer_type(model.parameters(), lr=lr)
        self.logger.info("Optimizer: %s", str(optimizer))
        if weight is None:
            loss = loss_type(reduction='mean')
            self.logger.info("Loss function: %s", str(loss))
        else:
            loss = loss_type(reduction='mean', weight=weight.to(self.device))
            self.logger.info("Loss function: %s with class_weights", str(loss))
        history = {'val_metrics': []}
        metric_name = metric.__name__ if hasattr(metric, "__name__") else 'metric'
        self.len_trainset = 0
        if metric_during_epoch:
            history['train_metrics'] =  []
            self.logger.info("Calculating train_metric: metric '%s' on train during epoch", metric_name)
            self.len_trainset = len(trainloader.dataset)
        if trainvalloader is not None:
            history['trainval_metrics'] =  []
            self.logger.info("Calculating trainval_metric: metric '%s' after each epoch", metric_name)
        self.logger.info("Calculating val_metric: metric '%s' (after each epoch)", metric_name)


        best_metric = 0
        best_epoch = None
        for epoch in range(1, 1 + epochs):
            self.logger.info("\n")
            self.logger.info(f"{epoch=}")
            self.train_epoch(model, optimizer, trainloader, loss, metric=metric) # training

            # Getting metrics:
            if metric_during_epoch:
                self.train_metric /= self.len_trainset
                assert self.total_len == self.len_trainset
                history['train_metrics'].append(self.train_metric)

            if trainvalloader is not None:
                metric_on_trainval = self.evaluation(model, trainvalloader, metric=metric)
                history['trainval_metrics'].append(metric_on_trainval)

            metric_on_val = self.evaluation(model, valloader, metric=metric)
            history['val_metrics'].append(metric_on_val)

            if 'train_metrics' in history:
                self.logger.info(f"Train (during): {metric_name} = {self.train_metric:.3f}")
            if 'trainval_metrics' in history:
                self.logger.info(f"Train (after):  {metric_name} = {metric_on_trainval:.3f}")
            if 'val_metrics' in history:
                self.logger.info(f"Validation:     {metric_name} = {metric_on_val:.3f}")

            # Compare val_metric with the best:
            if metric_on_val > best_metric:
                best_metric = metric_on_val
                best_epoch = epoch
                torch.save(model.state_dict(), self.best_checkpoint)
        self.logger.info(f"Best {metric_name} on val: {best_metric:.3f}")
        self.logger.info(f"Best checkpoint with this metric was saved to '%s' at epoch %d", self.best_checkpoint, best_epoch)
        return history

