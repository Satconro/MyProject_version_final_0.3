# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from uiqm_utils import getUIQM
from tqdm import tqdm



def calculate_metrics_ssim_psnr(generated_image_path, ground_truth_image_path, resize_size=(256, 256)):
    print("calculate_metrics_ssim_psnr")
    generated_image_list = os.listdir(generated_image_path)
    error_list_ssim, error_list_psnr = [], []

    for img in tqdm(generated_image_list):
        label_img = img
        generated_image = os.path.join(generated_image_path, img)
        ground_truth_image = os.path.join(ground_truth_image_path, label_img)

        generated_image = cv2.imread(generated_image)
        generated_image = cv2.resize(generated_image, resize_size)

        ground_truth_image = cv2.imread(ground_truth_image)
        ground_truth_image = cv2.resize(ground_truth_image, resize_size)

        # calculate SSIM
        error_ssim, diff_ssim = structural_similarity(generated_image, ground_truth_image, full=True, multichannel=True)
        error_list_ssim.append(error_ssim)

        generated_image = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)
        ground_truth_image = cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2GRAY)

        # calculate PSNR
        error_psnr = peak_signal_noise_ratio(generated_image, ground_truth_image)
        error_list_psnr.append(error_psnr)

    return np.array(error_list_ssim), np.array(error_list_psnr)

def calculate_UIQM(image_path, resize_size=(256, 256)):
    print("calculate_UIQM")
    image_list = os.listdir(image_path)
    uiqms = []

    for img in tqdm(image_list):
        image = os.path.join(image_path, img)

        image = cv2.imread(image)
        image = cv2.resize(image, resize_size)

        # calculate UIQM
        uiqms.append(getUIQM(image))
    return np.array(uiqms)


if __name__ == "__main__":
    # UIEBD = ["/home/lsm/home/experiment&result/vfinal_0.3/confidence/epoch_90/UIEB/raw-890",
    #          "/home/lsm/home/datasets/UIEB/reference-890"]
    # EUVP_dark = ["/home/lsm/home/experiment&result/vfinal_0.3/confidence/epoch_90/EUVP/underwater_dark",
    #              "/home/lsm/home/datasets/EUVP/Paired/underwater_dark/trainB"]
    # UFO_120 = ["/home/lsm/home/experiment&result/vfinal_0.3/confidence/epoch_90/UFO-120",
    #            "/home/lsm/home/datasets/UFO-120/TEST/hr"]

    generated_image_path = "/home/lsm/home/experiment&result/vfinal_0.3/confidence/epoch_90/DUO"
    # ground_truth_image_path = "/home/lsm/home/datasets/UFO-120/TEST/hr"
    # ssims, psnrs = calculate_metrics_ssim_psnr(generated_image_path, ground_truth_image_path)
    uiqms = calculate_UIQM(generated_image_path)
    # print("PSNR:{:.2f}±{:.2f}".format(np.mean(psnrs), np.std(psnrs)))
    # print("SSIM:{:.2f}±{:.2f}".format(np.mean(ssims), np.std(ssims)))
    print("UIQM:{:.2f}±{:.2f}".format(np.mean(uiqms), np.std(uiqms)))
    pass