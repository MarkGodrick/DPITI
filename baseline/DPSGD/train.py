import torch
from torch import nn, optim
from diffusers import UNet2DConditionModel, DDPMScheduler
from datasets import load_dataset
from opacus import PrivacyEngine
from opacus.optimizers import DPOptimizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class PrivacyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return torch.tensor(data['image']), torch.tensor(data['label'])

model = UNet2DConditionModel.from_pretrained("CompVis/ldm-3-5-256")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

scheduler = DDPMScheduler.from_config("CompVis/ldm-3-5-256")

dataset = load_dataset("your_privacy_dataset_name")
privacy_dataset = PrivacyDataset(dataset['train'])

train_dataloader = DataLoader(privacy_dataset, batch_size=32, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

privacy_engine = PrivacyEngine(
    model,
    batch_size=32,
    sample_size=len(dataset['train']),
    alphas=[10, 100],
    noise_multiplier=1.1,  # 控制噪声强度
    max_grad_norm=1.0,     # 梯度裁剪的最大值
)
privacy_engine.attach(optimizer)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        noise = torch.randn_like(images)  # 模拟噪声
        noisy_images = scheduler.add_noise(images, noise)
        outputs = model(noisy_images)

        # 计算损失 (假设是L2损失)
        loss = nn.MSELoss()(outputs, images)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    epsilon, best_alpha = privacy_engine.get_privacy_spent(steps=epoch)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item()} - Privacy: ε={epsilon}, α={best_alpha}")

model.save_pretrained("finetuned_diffusion_model")
