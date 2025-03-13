import os
import numpy as np
from torchvision.datasets import LSUN
from torchvision import transforms
from cleanfid.inception_torchscript import InceptionV3W
from cleanfid.resize import build_resizer
from PIL import Image
import torch
from tqdm import tqdm
import cleanfid.fid


np.random.seed(42)
num_samples = 20000
IMAGE_SIZE = 256
BATCH_SIZE = 32
SAVE_PATH = f"dataset/lsun/embedding/length_{num_samples:08}"
transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),transforms.CenterCrop(IMAGE_SIZE)])
inception = InceptionV3W(path="/data/whx/models", download=True, resize_inside=False).to("cuda")
resizer = build_resizer("clean")


dataset = LSUN(root="dataset/lsun",classes=["bedroom_train"],transform=transform)
indices_1 = np.arange(num_samples)
indices_2 = np.random.choice(len(dataset),num_samples)
indices_3 = np.random.choice(len(dataset),num_samples)
indices_list = [indices_1,indices_2,indices_3]
# compute embeddings of indices_1
emb_list_for_indices = []
os.makedirs(SAVE_PATH,exist_ok=True)

for num_idx, indices in enumerate(indices_list):
    
    if os.path.exists(os.path.join(SAVE_PATH,f"indices_{num_idx}.npy")):
        embedding_list = np.load(os.path.join(SAVE_PATH,f"indices_{num_idx}.npy"))
        mu = np.mean(embedding_list, axis=0)
        sigma = np.cov(embedding_list, rowvar=False)
        emb_list_for_indices.append((mu,sigma))
        continue

    embedding_list = []
    for batch_ptr in tqdm(range(0,num_samples,BATCH_SIZE)):
        images = []
        for idx in range(batch_ptr,min(num_samples,batch_ptr+BATCH_SIZE)):
            image,_ = dataset[indices[idx]]
            assert isinstance(image,Image.Image)
            image = resizer(np.array(image))
            images.append(image)
        images = np.array(images).transpose(0,3,1,2)
        assert images.shape[1]==3
        assert images.dtype==np.float32
        embeddings = inception(torch.from_numpy(images).to("cuda"))
        embedding_list.append(embeddings)
    embedding_list = torch.cat(embedding_list,dim=0)
    embedding_list = embedding_list.cpu().detach().numpy()
    np.save(os.path.join(SAVE_PATH,f"indices_{num_idx}"),embedding_list)
    mu = np.mean(embedding_list, axis=0)
    sigma = np.cov(embedding_list, rowvar=False)
    emb_list_for_indices.append((mu,sigma))

fid1 = cleanfid.fid.frechet_distance(
    emb_list_for_indices[1][0],
    emb_list_for_indices[1][1],
    emb_list_for_indices[0][0],
    emb_list_for_indices[0][1]
)

fid2 = cleanfid.fid.frechet_distance(
    emb_list_for_indices[1][0],
    emb_list_for_indices[1][1],
    emb_list_for_indices[2][0],
    emb_list_for_indices[2][1]
)

print(f"FID value for previous {num_samples} samples and random {num_samples} samples is {fid1}")
print(f"FID value for random {num_samples} samples and random {num_samples} samples is {fid2}")