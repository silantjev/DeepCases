from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

LEVELS2FROZEN = (0, 2, 5, 6, 7, 8)


def prepare_resnet(model, out_dim, logger=None, frozen=0):
    if frozen not in LEVELS2FROZEN:
        if logger is None:
            print(f"Argument frozen is {frozen}. Admissible values: {LEVELS2FROZEN}")
        else:
            logger.error(f"Argument frozen is {frozen}. Admissible values: {LEVELS2FROZEN}")
        raise ValueError()

    # Заменим последний слой
    model.fc = nn.Linear(
            in_features=model.fc.in_features,
            out_features=out_dim,
            bias=True,
        )
    # Поменяем инициализацию последнего слоя:
    try:
        nn.init.normal_(model.fc.weight, std=0.01)
        nn.init.constant_(model.fc.bias, 0)
    except Exception as e:
        if logger is None:
            print(f"Error while nn.init.kaiming_uniform_: {e}")
        else:
            logger.error(f"Error while nn.init.kaiming_uniform_: {e}")

    # Заморозим flozen первых уровней
    for i, child in enumerate(model.children()):
        if i >= frozen:
            break
        for param in child.parameters():
            param.requires_grad = False
        child.eval()
    
    return model

def make_resnet(logger, n_classes, frozen):
    weights = ResNet50_Weights.IMAGENET1K_V2
    logger.debug(f"Loading weights \"%s\"...", weights)
    model = resnet50(weights=weights)
    model = prepare_resnet(model, out_dim=n_classes, logger=logger, frozen=frozen)
    logger.debug("Weights loaded")
    return model
