import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

class MVTecDataset(Dataset):
    def __init__(self, dataset_path, class_name='bottle', is_train=True, resize=256, cropsize=224, wild_ver=False):
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        self.x, self.y, self.mask = self.load_dataset_folder()

        if wild_ver:
            self.transform_x =   T.Compose([T.Resize(resize, InterpolationMode.LANCZOS),
                                            T.RandomRotation(10),
                                            T.RandomCrop(cropsize),
                                            T.ToTensor(),
                                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])

            self.transform_mask =    T.Compose([T.Resize(resize, InterpolationMode.NEAREST),
                                                T.RandomRotation(10),
                                                T.RandomCrop(cropsize),
                                                T.ToTensor()])

        else:
            self.transform_x =   T.Compose([T.Resize(resize, InterpolationMode.LANCZOS),
                                            T.CenterCrop(cropsize),
                                            T.ToTensor(),
                                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])

            self.transform_mask =    T.Compose([T.Resize(resize, InterpolationMode.NEAREST),
                                                T.CenterCrop(cropsize),
                                                T.ToTensor()])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if y == 0 or mask is None:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                    for f in os.listdir(img_type_dir)
                                    if f.endswith(('.png', '.jpg', '.jpeg'))])
            x.extend(img_fpath_list)

            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                for img_fname in img_fname_list:
                    gt_path = os.path.join(gt_type_dir, img_fname + '_mask.png')
                    gt_path_jpg = os.path.join(gt_type_dir, img_fname + '_mask.jpg')
                    if os.path.exists(gt_path):
                        mask.append(gt_path)
                    elif os.path.exists(gt_path_jpg):
                        mask.append(gt_path_jpg)
                    else:
                        mask.append(None)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)
