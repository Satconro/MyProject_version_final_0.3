import os
import random
import numpy as np
import torch


def save_checkpoint(model, optimizer, filename):
    print("=> Saving checkpoint {}".format(filename))
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, device, optimizer=None, lr=None):
    print("=> Loading checkpoint from {}".format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if lr is not None:
        # If we don't do this then it will just have learning rate of old checkpoint
        # and it will lead to many hours of debugging \:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mkdirs(dir_path):
    if isinstance(dir_path, list):
        for dir in dir_path:
            if not os.path.exists(dir):
                os.makedirs(dir)
    else:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


def find_img(root, relative_root=''):
    # 递归地找到一个文件夹下的所有图片路径，返回该文件夹的绝对路径，和其下的所有图片在该文件夹下的相对路径
    if os.path.isdir(root):
        sub_list = []
        for i in os.listdir(root):
            sub_list.extend(find_img(os.path.join(root, i), i))
        return [os.path.join(relative_root, i) for i in sub_list]
    else:
        return [os.path.join(root.split(os.sep)[-1])]


def add_project_root_to_python_path():
    # 路径及sys.path处理
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[0]  # 从当前文件路径获取项目根路径
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.extend([str(PROJECT_ROOT)])  # 添加项目根路径到pythonPath
