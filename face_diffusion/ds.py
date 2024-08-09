import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, img_size=256, transform=None):
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not self.transform:
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


if __name__ == "__main__":
    img_dir = "/root/prakash/data/ffhq/kaggle-images/"
    ds = ImageFolderDataset(img_dir)
    x = ds[0]
    print(x.shape)