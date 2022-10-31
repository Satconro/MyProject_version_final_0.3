import os
import random

from torchvision import transforms

from datasets import base as Base


# def get_EUVP_Dataset(root, sub_set, img_size, num=None):
#     trans = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Resize([img_size, img_size]),
#     ])
#     sub_path = os.path.join(root, "Paired", sub_set)
#     assert os.path.exists(sub_path), "{}".format(sub_path)
#     root_d = os.path.join(sub_path, "trainA")
#     root_c = os.path.join(sub_path, "trainB")
#     if num is not None:
#         return Base.Paired_Image_Dataset(root_d, root_c, trans).squeeze(num)
#     else:
#         return Base.Paired_Image_Dataset(root_d, root_c, trans)


def get_EUVP_Dataset(root, img_size, num=None):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([img_size, img_size]),
    ])
    root_d = os.path.join(root, "trainA")
    root_c = os.path.join(root, "trainB")
    if num is not None:
        return Base.Paired_Image_Dataset(root_d, root_c, trans).squeeze(num)
    else:
        return Base.Paired_Image_Dataset(root_d, root_c, trans)


def get_DUO_Dataset(root, img_size, num=None):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([img_size, img_size])
    ])
    dataset = Base.Image_Dataset(root, trans)
    if num is not None:
        dataset.squeeze(num)
    return dataset


def get_RUIE_Dataset(root, img_size):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([img_size, img_size]),
    ])
    dataset = Base.Image_Dataset(root, trans)
    return dataset




# class UIEBD_DatasetBase(Paired_Image_Dataset):
#     def __init__(self, root):
#         """
#             读取UIEB数据集下的所有数据，针对不同分辨率的图像采取不同的裁剪操作，如果输入图像尺寸大于256*256，采用随机裁剪，否则resize为256*256
#         :param root: 文件根目录
#         """
#         self.root = root
#         super(UIEBD_DatasetBase, self).__init__(
#             root_d=os.path.join(root, "raw-890"),
#             root_c=os.path.join(root, "reference-890"),
#         )
#         self.transform_1 = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Resize([256, 256]),  # 图像增强执行随机翻转的意义不大
#         ])
#         self.transform_2 = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.RandomCrop([256, 256])
#             # transforms.RandomCrop(256, 256)
#         ])
#
#     # 重写__getitem__方法，修改格式转换操作
#     def __getitem__(self, index):
#         # 根据索引号读取图像数据，
#         name_1, img_1 = read_img(self.root_d, self.imgs_d[index % self.len])
#         name_2, img_2 = read_img(self.root_c, self.imgs_c[index % self.len])
#         assert name_1 == name_2
#
#         imgs = []
#         for img in [img_1, img_2]:
#             img_size = img.size
#             if not (img_size[0] > 256 and img_size[1] > 256):
#                 img = self.transform_1(img)
#                 imgs.append(img)
#             else:
#                 img = self.transform_2(img)
#                 imgs.append(img)
#         return name_1, imgs[0], imgs[1]


def test():
    pass


if __name__ == '__main__':
    test()
