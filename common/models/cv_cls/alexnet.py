from collections import OrderedDict
import numpy as np
from torch import nn

IM_SIZE = 128

class ConvBlock(nn.Module):
    """ Свёртку с соседними слоями и функцией активации выдилим в отдельный блок """
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size,  # для Conv
                 maxpooling=0, # есть ли MaxPool и сколько его kernel_size
                 activation_type=nn.ReLU,
                 padding='same', # для Conv
                 logger=None,
            ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_features)
        if maxpooling:
            self.pooling = nn.MaxPool2d(kernel_size=maxpooling)
        self.act = activation_type()
        # Подстроим инициализацию свёрточного слоя под пременяемую функцию активации:
        try:
            nn.init.kaiming_uniform_(self.conv.weight, nonlinearity=activation_type.__name__.lower())
            nn.init.constant_(self.conv.bias, 0)
        except Exception as e:
            if logger is None:
                print(f"Error while doing nn.init.kaiming_uniform_: {e}")
            else:
                logger.error(f"Error while doing nn.init.kaiming_uniform_: {e}")
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if hasattr(self, 'pooling'):
            x = self.pooling(x)
        x = self.act(x)
        return x


def create_cnn(logger=None, conv_feats=None, fc_feats=512,
                    out_dim=102, im_size=IM_SIZE):
    if conv_feats is None:
        conv_feats = (128, 128, 256, 256, 512, 512)
    assert len(conv_feats) == 6
    input_size = np.array((im_size, im_size))
    print(f"last conv output shape: {input_size}")
    input_size = input_size // 2 # L1
    input_size = input_size // 2 # L2
    input_size = input_size // 2 # L3
    flatten_output_dim = input_size.prod() * conv_feats[-1]
    print(f"last conv output shape: {input_size}")
    print(f"{flatten_output_dim=}")
    
    model = nn.Sequential(OrderedDict({
        'L1C1': ConvBlock(in_features=3, out_features=conv_feats[0],
                        kernel_size=3, logger=logger),
        'L1C2': ConvBlock(in_features=conv_feats[0], out_features=conv_feats[1],
                        kernel_size=3, maxpooling=2, logger=logger),
        'D1': nn.Dropout2d(0.2),
        'L2C1': ConvBlock(in_features=conv_feats[1], out_features=conv_feats[2],
                        kernel_size=3, logger=logger),
        'L2C2': ConvBlock(in_features=conv_feats[2], out_features=conv_feats[3],
                        kernel_size=3, maxpooling=2, logger=logger),
        'D2': nn.Dropout2d(0.25),
        'L3C1': ConvBlock(in_features=conv_feats[3], out_features=conv_feats[4],
                        kernel_size=3, logger=logger),
        'L3C2': ConvBlock(in_features=conv_feats[4], out_features=conv_feats[5],
                        kernel_size=3, maxpooling=2, logger=logger),
        'D3': nn.Dropout2d(0.25),
        'FLAT': nn.Flatten(),
        'FC1': nn.Sequential(
                    nn.Linear(in_features=flatten_output_dim, out_features=fc_feats),
                    nn.ReLU(),
                ),
        'FC2': nn.Sequential(
                    nn.Dropout(0.25),
                    nn.Linear(in_features=fc_feats, out_features=out_dim),
                ),
    }))
    # Поменяем инициализацию последнего слоя:
    try:
        nn.init.normal_(model.FC2[1].weight, std=0.01)
        nn.init.constant_(model.FC2[1].bias, 0)
    except Exception as e:
        if logger is None:
            print(f"Error while nn.init.kaiming_uniform_: {e}")
        else:
            logger.error(f"Error while nn.init.kaiming_uniform_: {e}")

    return model

