import logging
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def compare_on_plot(history, name, metric_name='metric', logger='plot', save_path=None):
    """ Строит график для метрик для train и val """
    if isinstance(logger, str):
        logger = logging.getLogger(logger)
        logger.addHandler(logging.StreamHandler())

    train = history.get('train_metrics') # метрика времени обучения
    trainval = history.get('trainval_metrics') # в конце эпохи
    val = history.get('val_metrics')
    if trainval is not None and val is not None and (trainval.shape == val.shape) and (trainval == val).all():
        logger.warning("trainval and train coincide")

    if train is None and trainval is None and val is None:
        message = "[compare_on_plot] Not train nor trainval nor val in histroy"
        logger.error(message)
        return

    plt.style.use(['dark_background'])
    fig, ax = plt.subplots(figsize=(10, 5))
    if train is not None:
        epochs = np.arange(1, len(train) + 1) - 0.5
        plt.plot(epochs, train, label='Train (during epoch)')
    if trainval is not None:
        epochs = range(1, len(trainval) + 1)
        plt.plot(epochs, trainval, label='Train (after epoch)')
    if val is not None:
        epochs = range(1, len(val) + 1)
        plt.plot(epochs, val, label='Validation')
    plt.title(f'Model {name}')
    plt.ylabel(metric_name)
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    if save_path is not None:
        plt.savefig(str(save_path))
        message = f'Training history plot is saved to {save_path}'
        logger.info(message)
        plt.clf()
    else:
        plt.show()

def show_evaluation_results(all_labels, all_preds, classes):
    print(classification_report(all_labels, all_preds, target_names=classes))

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()  
