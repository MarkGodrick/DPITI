import os
import sys
import argparse
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, Subset
from torchvision.datasets import LSUN
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3
from torchvision import transforms
from torch.utils.data import DataLoader
from scipy.linalg import sqrtm
from tqdm import tqdm
from logger import execution_logger, setup_logging


transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Resize((299, 299)), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

class Images(Dataset):
    def __init__(self,data):
        super().__init__()
        self.isdataset = isinstance(data,Dataset)
        self.dataset = data

    def __len__(self):
        return min(20000,len(self.dataset))
    
    def __getitem__(self,idx):
        if self.isdataset:
            return self.dataset[idx][0]
            # return np.array(self.dataset[idx][0])
        else:
            return transform(self.dataset[idx])


# ---- 1. 计算 Inception 网络的特征 ---- #
def get_inception_model(device):
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.fc = nn.Identity()  # 移除全连接层，只取倒数第二层的特征
    model.eval()
    return model


# ---- 2. 计算数据集的 Inception 统计特征 (均值和协方差) ---- #
def compute_statistics(images, model, device, batch_size=32):
    """
    计算数据集的 Inception 特征均值和协方差
    images: list of numpy arrays or a PyTorch Dataset
    model: InceptionV3 model
    device: "cuda" or "cpu"
    batch_size: batch size for feature extraction
    """
    dataloader = DataLoader(images, batch_size=batch_size, shuffle=False)

    # 提取特征
    features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            batch = batch.to(device)
            feat = model(batch).cpu().numpy()
            features.append(feat)

    features = np.concatenate(features, axis=0)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)

    return mu, sigma


# ---- 3. 计算 FID (Fréchet Inception Distance) ---- #
def calculate_fid(mu1, sigma1, mu2, sigma2):
    """
    计算 FID 分数
    """
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)  # sqrtm: 计算协方差矩阵的平方根
    if np.iscomplexobj(covmean):
        covmean = covmean.real  # 可能存在小的虚部，去掉

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


# ---- 4. 计算 Inception Score (IS) ---- #
def calculate_inception_score(images, model, device, batch_size=32, splits=10):
    """
    计算 Inception Score (IS)
    """

    dataloader = DataLoader(images, batch_size=batch_size, shuffle=False)

    preds = []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing Inception Score"):
            batch = batch.to(device)
            logits = model(batch)
            prob = softmax(logits).cpu().numpy()
            preds.append(prob)

    preds = np.concatenate(preds, axis=0)
    
    scores = []
    chunk_size = preds.shape[0] // splits
    for i in range(splits):
        subset = preds[i * chunk_size:(i + 1) * chunk_size]
        p_y = np.mean(subset, axis=0)
        kl_div = subset * (np.log(subset) - np.log(p_y))
        scores.append(np.exp(np.mean(np.sum(kl_div, axis=1))))

    return np.mean(scores), np.std(scores)


def compute_fid_and_is(real_images, generated_images, device="cuda"):
    """
    计算 FID 和 Inception Score
    real_images: PyTorch Dataset 或 numpy.ndarray (真实图像)
    generated_images: numpy.ndarray (生成图像)
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 获取 Inception 模型
    model = get_inception_model(device)

    # 计算 FID 统计量
    execution_logger.info("Computing statistics for real images...")
    mu_real, sigma_real = compute_statistics(real_images, model, device)

    execution_logger.info("Computing statistics for generated images...")
    mu_gen, sigma_gen = compute_statistics(generated_images, model, device)

    execution_logger.info("Computing FID score...")
    fid = calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)

    execution_logger.info("Computing Inception Score...")
    inception_score, inception_std = calculate_inception_score(generated_images, model, device)

    return fid, inception_score, inception_std


def main(args):
    dataset = Images(LSUN(root=args.dataset,classes=['bedroom_train'],transform=transform))

    samples = np.load(os.path.join(args.input))
    samples = samples['arr_0']
    # samples = np.transpose(samples['arr_0'],axes=(0,3,1,2))
    execution_logger.info(f"input sample size:{samples.shape}")

    samples = Images(samples)
    # assert samples.shape==(14000,3,512,512)

    fid_score, is_score, is_std = compute_fid_and_is(dataset, samples, device="cuda")

    execution_logger.info(f"FID Score: {fid_score:.2f}")
    execution_logger.info(f"Inception Score: {is_score:.2f} ± {is_std:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",type=str,default='dataset/lsun')
    parser.add_argument("--input",type=str,default="results/image/LSUN_huggingface/part_0/huggingface")

    args = parser.parse_args()

    setup_logging(log_file=os.path.join(os.path.dirname(args.input),"eval_log.txt"))

    execution_logger.info("\nExecuting {}...\ndataset: {}\ninput: {}".format(sys.argv[0],args.dataset,args.input))

    main(args)