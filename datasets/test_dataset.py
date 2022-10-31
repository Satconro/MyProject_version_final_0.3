import utils
import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


class Test_Dataset(Dataset):
    def __init__(self, root):
        super(Test_Dataset, self).__init__()
        self.root = root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize([256, 256])
        ])

        self.imgs = utils.find_img(self.root)
        self.len = len(self.imgs)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img_name, img_path = self.imgs[index], os.path.join(self.root, self.imgs[index])
        img = self.transform(Image.open(img_path))
        img_size = img.shape
        if img_size[1] * img_size[2] > 1920 * 1080:
            trans = transforms.Resize([img_size[1] // 2, img_size[2] // 2])
            img = trans(img)
        return img_name, img
