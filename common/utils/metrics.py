from sklearn.metrics import f1_score

def f1_macro(gt, pred):
    return f1_score(gt, pred, average="macro")

def accuracy(gt, pred):
    assert len(gt) == len(pred)
    return (gt == pred).mean()

