import os
from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
from torchvision import transforms
from torch.utils import data
from pathlib import Path
from itertools import chain


class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = sorted(data_utils.make_dataset(root))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image


class CelebaAttributeDataset(Dataset):
    def __init__(self, images_root, attr, transform=None, attributes_root="", use_attr=True):
        self.paths = data_utils.make_dataset(images_root)
        self.transform = transform
        with open(attributes_root, "r") as f:
            lines = f.readlines()

        attr_num = -1
        for i, data_attr in enumerate(lines[1].split(" ")):
            if data_attr.strip() == attr.strip():
                attr_num = i
                break
        assert attr_num > -1, f"Can not find attribute {attr}"

        filtred_paths = []
        for path in self.paths:
            pic_num = int(path.split("/")[-1].replace(".jpg", "").replace(".png", ""))
            pic_attrs = lines[pic_num + 2].strip().split(" ")
            pic_attrs = pic_attrs[2:]
            if use_attr and pic_attrs[attr_num] == "1" or not use_attr and pic_attrs[attr_num] == "-1":
                filtred_paths.append(path)
        self.paths = sorted(filtred_paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from_path = self.paths[index]
        image = Image.open(from_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image


class FIDDataset(Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        file = self.files[i]
        image = file.convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        return image


class MetricsPathsDataset(Dataset):
    def __init__(self, root_path, gt_dir=None, transform=None, transform_train=None, return_path=False, ignore=[]):
        self.pairs = []
        self.paths = []
        self.names = []

        for f in os.listdir(root_path):
            if f not in ignore:
                self.names.append(f)
                image_path = os.path.join(root_path, f)
                gt_path = os.path.join(gt_dir, f)
                if f.endswith(".jpg") or f.endswith(".png"):
                    self.pairs.append([image_path, gt_path.replace(".png", ".jpg"), None])
                    self.paths.append(image_path)
        self.transform = transform
        self.transform_train = transform_train
        self.return_path = return_path

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        from_path, to_path, _ = self.pairs[index]
        from_im = Image.open(from_path).convert("RGB")
        to_im = Image.open(to_path).convert("RGB")

        if self.transform:
            to_im = self.transform(to_im)
            from_im = self.transform(from_im)

        if not self.return_path:
            return from_im, to_im
        else:
            return from_im, to_im, self.names[index]


class MetricsDataDataset(Dataset):
    def __init__(
        self, paths, target_data, fake_data, transform=None, transform_train=None
    ):
        self.fake_data = fake_data
        self.target_data = target_data
        self.paths = paths
        self.transform = transform
        self.transform_train = transform_train

    def __len__(self):
        return len(self.fake_data)

    def __getitem__(self, index):

        target_im = self.target_data[index]
        fake_im = self.fake_data[index]

        if self.transform:
            fake_im = self.transform(fake_im)
            target_im = self.transform(target_im)

        return target_im, fake_im

def get_eval_loader(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False):
    print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.Resize([height, width]),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        transform = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.Resize([height, width]),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    dataset = DefaultDataset(root, transform=transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)

def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames

class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]

        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)