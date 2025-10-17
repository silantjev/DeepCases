import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T

class RandomCropper:
    def __init__(self, max_cut=0.15):
        assert 2*max_cut < 1
        self.max_cut = max_cut
    
    def __call__(self, x):
        # if not isinstance(x, np.ndarray):
            # x = np.array(x)
        shape = x.shape
        assert len(shape) == 3
        assert shape[2] == 3
        h, w = shape[:2]
        cuts = np.array([h, h, w, w], dtype=np.float64)
        cuts *= np.random.random(4) * self.max_cut
        top, bottom, left, right = cuts.astype(int) 
        return x[top:w-bottom, left:w-right]

class TorchRandomCropper:
    def __init__(self, max_cut=0.15):
        assert 2*max_cut < 1
        self.max_cut = max_cut
    
    def __call__(self, x):
        shape = x.shape
        assert len(shape) == 3
        assert shape[0] == 3
        h, w = shape[1:]
        cuts = torch.tensor([h, h, w, w], dtype=torch.float32)
        cuts *= np.random.random(4) * self.max_cut
        top, bottom, left, right = cuts.to(dtype=torch.int) 
        return x[:, top:w-bottom, left:w-right]

class PreTransforms:
    def __init__(self, im_size, swapRGB, max_cut=0):
        self.im_size = im_size
        self.swapRGB = swapRGB
        if max_cut:
            self.cropper = TorchRandomCropper(max_cut)
            self.resizer = T.Resize((self.im_size, self.im_size), antialias=True)

    def preprocess(self, x):
        # if hasattr(self, "cropper"):
            # x = self.cropper(x)
        x = cv2.resize(x, (self.im_size, self.im_size))
        x = x.transpose((2, 0, 1))
        if self.swapRGB:
            x = x[::-1]
        x = np.ascontiguousarray(x)
        return x

    def common_transformations(self, x):
        return torch.tensor(x / 255.0, dtype=torch.float32)

    def crop(self, x):
        if hasattr(self, "cropper"):
            x = self.cropper(x)
            x = self.resizer(x)
        return x

class FlowerDataSet(Dataset):
    def __init__(self, paths, indices, labels, im_size,
                 transform=None, device='cpu', load_images=True, max_cut=0):
        self.device = torch.device(device)
        self.im_size = im_size
        self.pretransforms = PreTransforms(im_size=im_size, swapRGB=True, max_cut=max_cut)
        n = len(labels)
        assert len(paths) >= indices.max() >= len(indices) == n
        self.transform = transform
        self.y = torch.LongTensor(labels).to(self.device)
        
        if load_images:
            self.X = []
            for i in indices:
                path = paths[i]
                assert path.exists()
                image = cv2.imread(str(path))
                image = self.pretransforms.preprocess(image)
                self.X.append(image)
                del image
        else:
            self.paths = []
            for i in indices:
                path = paths[i]
                assert path.exists()
                self.paths.append(path)
    
    def __getitem__(self, idx):
        if hasattr(self, 'X'):
            image = self.X[idx]
        else:
            image = cv2.imread(str(self.paths[idx]))
            image = self.pretransforms.preprocess(image)
        image = self.pretransforms.common_transformations(image)
        image = self.pretransforms.crop(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.y[idx]
        
    def __len__(self):
        return len(self.y)

