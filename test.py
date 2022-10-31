from utils import add_project_root_to_python_path
add_project_root_to_python_path()
import os
import argparse
from tqdm import tqdm
import utils
from models.model import Encoder, Decoder
from datasets.test_dataset import Test_Dataset

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image


class Enhancement_Model_Tester:
    def __init__(self, running_config):
        # Config
        self.config = running_config
        # Model: load tested model and set mode to eval()
        self.encoder = Encoder().to(self.config.device).eval()
        self.decoder = Decoder().to(self.config.device).eval()
        # Data:
        self.dataset = Test_Dataset(root=self.config.root)
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=0,
        )
        self._load_model()
        torch.set_grad_enabled(False)

    def _load_model(self):
        enc_path = os.path.join(self.config.snapshots_folder, 'encoder.ckpt')
        dec_path = os.path.join(self.config.snapshots_folder, 'decoder.ckpt')
        utils.load_checkpoint(enc_path, self.encoder, self.config.device)
        utils.load_checkpoint(dec_path, self.decoder, self.config.device)

    def test(self):
        pbar = tqdm(enumerate(self.dataloader),
                    total=len(self.dataloader),
                    unit='batch',
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                    ncols=150)
        for batch_n, (img_path, img) in pbar:
            with torch.no_grad():
                img = img.to(self.config.device)
                embedding, enc_outs = self.encoder(img)
                # x1, x2, x3, x4, x_bat = enc_outs
                # zero_x1 = torch.zeros_like(x1)
                # zero_x2 = torch.zeros_like(x2)
                # zero_x3 = torch.zeros_like(x3)
                # zero_x4 = torch.zeros_like(x4)
                # zero_x_bat = torch.zeros_like(x_bat)
                generated_imgs = self.decoder(enc_outs)
                # Save img
                if not os.path.exists(self.config.target_dir):
                    utils.mkdirs(self.config.target_dir)
                if not os.path.exists(os.path.join(self.config.target_dir, *img_path[0].split(os.sep)[0:-1])):
                    utils.mkdirs(os.path.join(self.config.target_dir, *img_path[0].split(os.sep)[0:-1]))
                print(img_path[0])
                save_image(generated_imgs, os.path.join(self.config.target_dir, img_path[0]))
                # del img
                # del generated_imgs
                # torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, default=r"/home/lsm/home/datasets/UFO-120/TEST/lrd")
    parser.add_argument('--target_dir', type=str, default=r"/home/lsm/home/experiment&result/vfinal_0.3/confidence_no_AD/epoch_100/UFO-120")
    parser.add_argument('--snapshots_folder', type=str, default="/home/lsm/home/snapshots/final_0.3/confidence_no_AD/epoch_100")

    config = parser.parse_args()
    # config.device = "cpu"
    # config.device = "cuda"
    # torch.cuda.set_device(2)
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tester = Enhancement_Model_Tester(config)
    tester.test()
