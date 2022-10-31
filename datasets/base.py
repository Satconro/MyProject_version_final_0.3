import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def find_img(root, relative_root=''):
    # 递归地找到一个文件夹下的所有图片路径，返回该文件夹的绝对路径，和其下的所有图片在该文件夹下的相对路径
    if os.path.isdir(root):
        sub_list = []
        for i in os.listdir(root):
            sub_list.extend(find_img(os.path.join(root, i), i))
        return [os.path.join(relative_root, i) for i in sub_list]
    else:
        return [os.path.join(root.split(os.sep)[-1])]


def read_img(root, img):
    return img, Image.open(os.path.join(root, img))


def get_imgs(root, relative_root=''):
    """
    递归地找到一个文件夹下的所有图片路径，返回该文件夹的绝对路径，和其下的所有图片在该文件夹下的相对路径
    """
    if os.path.isdir(root):
        sub_list = []
        for i in os.listdir(root):
            sub_list.extend(get_imgs(os.path.join(root, i), i))
        return [os.path.join(relative_root, i) for i in sub_list]
    else:
        return [os.path.join(root.split(os.sep)[-1])]


class Image_Dataset(Dataset):
    def __init__(self, root, trans=None):
        super(Image_Dataset, self).__init__()
        self.root = root
        self.trans = trans
        if self.trans is None:
            self.trans = transforms.Compose([
                # 将PIL.Image转化为tensor，即归一化。 注：shape 会从(H，W，C)变成(C，H，W)；先ToTensor转换为Tensor才能进行正则化
                transforms.ToTensor(),
            ])
        # self.imgs = os.listdir(self.root)
        self.imgs = find_img(self.root)
        self.len = len(self.imgs)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img_name, img_path = self.imgs[index], os.path.join(self.root, self.imgs[index])
        img = Image.open(img_path)
        return img_name, self.trans(img)

    def squeeze(self, num):
        self.imgs = random.sample(self.imgs, num)
        self.len = len(self.imgs)


class Paired_Image_Dataset(Dataset):
    def __init__(self, root_d, root_c, trans=None):
        super(Paired_Image_Dataset, self).__init__()
        self.root_d = root_d
        self.root_c = root_c
        self.trans = trans
        if self.trans is None:
            self.trans = transforms.Compose([
                # 将PIL.Image转化为tensor，即归一化。 注：shape 会从(H，W，C)变成(C，H，W)；先ToTensor转换为Tensor才能进行正则化
                transforms.ToTensor(),
            ])
        self.imgs_d = os.listdir(self.root_d)  # 此处未转换为绝对路径
        self.imgs_c = os.listdir(self.root_c)
        assert len(self.imgs_d) == len(self.imgs_c), "The length of the raw images and the reference does not match."
        self.len = len(self.imgs_d)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # 根据索引号读取图像数据，
        name_1, img_1 = read_img(self.root_d, self.imgs_d[index % self.len])
        name_2, img_2 = read_img(self.root_c, self.imgs_c[index % self.len])
        return name_1, self.trans(img_1), self.trans(img_2)

    def squeeze(self, num):
        # 随机筛选指定个数的样本
        temp = list(zip(self.imgs_d, self.imgs_c))
        temp = random.sample(temp, num)
        self.imgs_d, self.imgs_c = zip(*temp)
        self.imgs_d = list(self.imgs_d)
        self.imgs_c = list(self.imgs_c)
        self.len = len(temp)


if __name__ == "__main__":
    root_RUIE = "/home/lsm/home/datasets/My_Dataset/RUIE"
    Image_Dataset(root_RUIE)