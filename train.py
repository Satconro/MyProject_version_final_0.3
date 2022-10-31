from utils import add_project_root_to_python_path

add_project_root_to_python_path()

import os
from pathlib import Path
import numpy
import argparse
from tqdm import tqdm
from itertools import cycle, chain

import torch
import torchvision
from torch.optim import Adam
from torch.utils.data import DataLoader

import utils
import datasets.my_dataset as my_dataset  # my_data
from models.model import Encoder, Decoder, Discriminator  # my_model
from losses.loss import EncoderLoss, DecoderLoss, DiscriminatorLoss  # my_loss


class Enhancement_Module_Trainer:
    def __init__(self, running_config):
        # Config:
        self.config = running_config
        # Dataset:
        self.data_real = my_dataset.get_RUIE_Dataset(self.config.root_RUIE, self.config.image_size)
        self.data_syn = my_dataset.get_EUVP_Dataset(self.config.root_EUVP, self.config.image_size)
        print("Datasets are loaded. "
              "length of real: {}, length of syn: {}".format(len(self.data_real), len(self.data_syn)))
        # Dataloader:
        self.dataloader_real = DataLoader(
            self.data_real,
            batch_size=running_config.batch_size,
            shuffle=True,  #CHANGE
            num_workers=2,
            pin_memory=True,
        )
        self.dataloader_synthetic = DataLoader(
            self.data_syn,
            batch_size=running_config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        # Model:
        self.encoder = Encoder().to(self.config.device).train(mode=True)
        self.decoder = Decoder().to(self.config.device).train(mode=True)
        self.discriminator = Discriminator(in_channels=64).to(self.config.device).train(mode=True)
        print("Models are loaded")
        # Optimizer & Loss:
        self.enc_opt = Adam(self.encoder.parameters(), lr=running_config.learning_rate)
        self.dec_opt = Adam(self.decoder.parameters(), lr=running_config.learning_rate)
        self.dis_opt = Adam(self.discriminator.parameters(), lr=running_config.learning_rate)
        self.vgg19 = torchvision.models.vgg19(pretrained=True).to(self.config.device).eval()
        print("VGG-19 model is loaded")
        self.enc_loss = EncoderLoss().to(self.config.device)
        self.dec_loss = DecoderLoss(self.vgg19, self.config.device).to(self.config.device)
        self.dis_loss = DiscriminatorLoss().to(self.config.device)
        # preTraining:
        self.pre_training = self.config.pre_training
        self.pre_threshold = self.config.pre_threshold

        if self.config.load_checkpoint is True:
            self.load_model()
            self.pre_training = False

    def load_model(self, epoch=None):
        print("load model")
        file_names = ['encoder.ckpt', 'decoder.ckpt', 'discriminator.ckpt']
        models = [
            (self.encoder, self.enc_opt),
            (self.decoder, self.dec_opt),
            (self.discriminator, self.dis_opt)
        ]
        for fileName, (model, opt) in zip(file_names, models):
            if epoch is not None:
                path = os.path.join(self.config.load_snapshots_folder, "epoch_" + str(epoch), fileName)
            else:
                path = os.path.join(self.config.load_snapshots_folder, fileName)
            utils.load_checkpoint(path, model, self.config.device, opt)

    def save_model(self, epoch=None):
        file_names = ['encoder.ckpt', 'decoder.ckpt', 'discriminator.ckpt']
        models = [
            (self.encoder, self.enc_opt),
            (self.decoder, self.dec_opt),
            (self.discriminator, self.dis_opt)
        ]
        for fileName, (model, opt) in zip(file_names, models):
            if epoch is not None:
                path = os.path.join(self.config.save_snapshots_folder, "epoch_" + str(epoch), fileName)
            else:
                path = os.path.join(self.config.save_snapshots_folder, fileName)
            if not os.path.exists(Path(path).resolve().parent):
                os.makedirs(Path(path).resolve().parent)
            utils.save_checkpoint(model, opt, path)

    def train_enhancement(self, synthetic, clear):
        # 更新编码器和解码器
        self.enc_opt.zero_grad()
        self.dec_opt.zero_grad()
        embedding, enc_outs = self.encoder(synthetic)
        generated_img = self.decoder(enc_outs)
        rec_loss = self.dec_loss(generated_img, clear)
        rec_loss.backward()
        self.enc_opt.step()
        self.dec_opt.step()
        return rec_loss.item()

    def train_dis(self, syn, real):
        # 更新鉴别器：对于每个batch的数据，dis需要多次更新
        self.dis_opt.zero_grad()
        r_embedding, _ = self.encoder(real)
        f_embedding, _ = self.encoder(syn)
        r_result = self.discriminator(r_embedding)
        f_result = self.discriminator(f_embedding)
        dis_loss = self.dis_loss(f_result, r_result, r_embedding, f_embedding, self.discriminator)
        dis_loss.backward()
        self.dis_opt.step()
        return dis_loss.item()

    def train_encoder(self, synthetic):
        # 更新编码器，编码器需要使生成的假图像的编码更真
        self.enc_opt.zero_grad()
        embedding, enc_outs = self.encoder(synthetic)
        result = self.discriminator(embedding)
        enc_loss = self.enc_loss(result)
        enc_loss.backward()
        self.enc_opt.step()
        return enc_loss.item()

    def pre_train(self, epoch, device):  # 在合成数据集上预先进行训练
        if self.pre_training:
            print("Start pre-training")
            for epoch_n in range(epoch):
                pbar = tqdm(enumerate(self.dataloader_synthetic),
                            total=len(self.dataloader_synthetic),
                            desc=f'Epoch {epoch_n + 1}/{epoch}',
                            unit='batch',
                            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                            ncols=150)
                sum_loss = 0
                for batch_n, (_, distorted, clear) in pbar:
                    distorted, clear = distorted.to(device), clear.to(device)
                    rec_loss = self.train_enhancement(distorted, clear)
                    sum_loss += rec_loss
                    pbar.set_postfix({'rec_loss': rec_loss, 'avg:': sum_loss / (batch_n + 1)})
                    if sum_loss <= self.pre_threshold:
                        self.save_model(epoch="pre_training")
                        return
            # 训练结束保存参数
            self.save_model(epoch="pre_training")

    def not_pre_train(self, epoch, device):  # 预训练完成后，交替训练两个网络
        # 记录训练过程
        # train_records = numpy.zeros(())
        for epoch_n in range(epoch):
            pbar = tqdm(enumerate(zip(self.dataloader_real, self.dataloader_synthetic)),
                        total=min(len(self.dataloader_real), len(self.dataloader_synthetic)),
                        desc=f'Epoch {epoch_n + 1}/{epoch}',
                        unit='batch',
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                        ncols=150
                        )
            sum_r_loss = sum_e_loss = sum_d_loss = 0
            for batch_n, data in pbar:
                _, real, _, syn, clear = chain.from_iterable(data)  # 解包
                if not len(syn) == len(real):  # 跳过batch_size不对齐的那一组数据
                    continue
                # Python 如何通过迭代修改对象
                real, syn, clear = real.to(device), syn.to(device), clear.to(device)

                # # 训练对抗性网络
                # for i in range(max(self.config.num_critic, 1)):
                #     dis_loss = self.train_dis(syn, real)
                #     sum_d_loss += dis_loss
                # enc_loss = self.train_encoder(syn)
                # sum_e_loss += enc_loss

                # 训练图像增强网络
                rec_loss = self.train_enhancement(syn, clear)
                sum_r_loss += rec_loss

                # 在进度条后显示当前batch的损失
                pbar.set_postfix({'average_e': sum_e_loss / (batch_n + 1),
                                  'average_d': sum_d_loss / ((batch_n + 1) * self.config.num_critic),
                                  'average_r': sum_r_loss / (batch_n + 1)})

            # 每10个epoch保存一次
            if (epoch_n + 1) % 10 == 0:
                self.save_model(epoch_n + 1)

    def train(self):
        self.pre_train(self.config.pre_training_epoch, self.config.device)
        self.not_pre_train(self.config.epoch, self.config.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--root_EUVP', type=str, default="/home/lsm/home/datasets/My_Dataset/EUVP_Paired")
    parser.add_argument('--root_RUIE', type=str, default="/home/lsm/home/datasets/My_Dataset/RUIE")
    # parser.add_argument('--root_DUO', type=str, default="/home/lsm/home/datasets/DUO/DUO/images/train")

    parser.add_argument('--pre_training', type=bool, default=False)
    parser.add_argument('--pre_training_epoch', type=int, default=10)
    parser.add_argument('--pre_threshold', type=float, default=0.14)

    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--num_critic', type=int, default=2)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.0001)

    parser.add_argument('--load_checkpoint', type=bool, default=False)
    parser.add_argument('--load_snapshots_folder', type=str, default="")

    parser.add_argument('--save_snapshots_folder', type=str, default="/home/lsm/home/snapshots/final_0.3/confidence_no_AD/")

    config = parser.parse_args()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = Enhancement_Module_Trainer(config)
    trainer.train()
