import os
import torch 
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

class ImageDL(Dataset):
    def __init__(self, image_dir, resolution, augmentation_transforms=None):
        """
        Args:
            image_dir (str): Directory with all the images.
            resolution (tuple): Desired resolution (height, width) to resize images.
            augmentation_transforms (torchvision.transforms.Compose, optional): 
                Transformations to be applied for data augmentation.
        """
        self.image_dir = image_dir
        self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            augmentation_transforms if augmentation_transforms else transforms.Lambda(lambda x: x),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image
    
def get_dl(image_dir, resolution, batch_size):
    augmentation_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ])
    dataset = ImageDL(image_dir, resolution, augmentation_transforms)
    dl = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4)
    return dl


# Example usage
if __name__ == "__main__":
    import numpy as np

    # Directory containing images
    image_dir = '/root/prakash/data/ffhq/thumbnails128x128/'
    
    # Desired resolution
    resolution = (128, 128)
    
    # Augmentation transforms (example)
    # augmentation_transforms = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(10),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    # ])

    # dataset = ImageDL(image_dir, resolution, augmentation_transforms)

    # # Display a sample image
    # sample_image = dataset[0]
    # sample_image = sample_image.permute(1, 2, 0).numpy()  # Convert from CHW to HWC
    # sample_image = np.uint8(255*((sample_image + 1) / 2))  # Convert from [-1, 1] to [0, 1]
    # Image.fromarray(sample_image).save("stylegan/check_ds.png")
    dl = get_dl(image_dir, resolution, batch_size=32)
    for images in dl:
        print(images.shape)