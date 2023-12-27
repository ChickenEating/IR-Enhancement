import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from models.DDP import DDPNet
from models.CIM import CIMNet
from utils.Metrics import PSNR, SSIM, VIF

# Define your dataset class
class IREDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.high_contrast_noisy_path = os.path.join(root_dir, "high_contrast_noisy")
        self.high_contrast_noise_free_path = os.path.join(root_dir, "high_contrast_noise_free")
        self.low_contrast_noisy_path = os.path.join(root_dir, "low_contrast_noisy")
        self.low_contrast_noise_free_path = os.path.join(root_dir, "low_contrast_noise_free")
        self.transform = transforms.Compose([transforms.ToTensor()])

        # Assuming all subfolders have the same set of filenames
        self.image_files = os.listdir(self.high_contrast_noisy_path)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path_high_contrast_noisy = os.path.join(self.high_contrast_noisy_path, image_name)
        image_path_high_contrast_noise_free = os.path.join(self.high_contrast_noise_free_path, image_name)
        image_path_low_contrast_noisy = os.path.join(self.low_contrast_noisy_path, image_name)
        image_path_low_contrast_noise_free = os.path.join(self.low_contrast_noise_free_path, image_name)

        image_high_contrast_noisy = Image.open(image_path_high_contrast_noisy).convert('L')
        image_high_contrast_noise_free = Image.open(image_path_high_contrast_noise_free).convert('L')
        image_low_contrast_noisy = Image.open(image_path_low_contrast_noisy).convert('L')
        image_low_contrast_noise_free = Image.open(image_path_low_contrast_noise_free).convert('L')

        image_high_contrast_noisy = self.transform(image_high_contrast_noisy)
        image_high_contrast_noise_free = self.transform(image_high_contrast_noise_free)
        image_low_contrast_noisy = self.transform(image_low_contrast_noisy)
        image_low_contrast_noise_free = self.transform(image_low_contrast_noise_free)

        return image_high_contrast_noisy, image_high_contrast_noise_free, image_low_contrast_noisy, image_low_contrast_noise_free

# Create an instance of your dataset
dataset = IREDataset(root_dir="./data/IRE")

# Split dataset into training and testing sets
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size
_, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loader for testing
batch_size = 32
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize your DDPNet and CIMNet models
ddp_model = DDPNet()
cim_model = CIMNet()

# Load trained models
ddp_model.load_state_dict(torch.load("ddp_model.pth"))
cim_model.load_state_dict(torch.load("cim_model.pth"))

# Set models to evaluation mode
ddp_model.eval()
cim_model.eval()

# Set up metrics for evaluation
psnr_metric = PSNR()
ssim_metric = SSIM()
vif_metric = VIF()

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    for test_batch_high_contrast_noisy, test_batch_high_contrast_noise_free, test_batch_low_contrast_noisy, test_batch_low_contrast_noise_free in test_loader:
        test_batch_high_contrast_noisy = test_batch_high_contrast_noisy.to(device)
        test_batch_high_contrast_noise_free = test_batch_high_contrast_noise_free.to(device)
        test_batch_low_contrast_noisy = test_batch_low_contrast_noisy.to(device)
        test_batch_low_contrast_noise_free = test_batch_low_contrast_noise_free.to(device)

        # DDPNet forward pass
        test_denoised_image, _, test_detail_feautre = ddp_model(test_batch_low_contrast_noisy)

        # CIMNet forward pass
        test_enhanced_image = cim_model(test_denoised_image, test_detail_feautre)

        # Convert tensors to numpy arrays
        denoised_np = test_denoised_image.cpu().numpy().squeeze()
        enhanced_np = test_enhanced_image.cpu().numpy().squeeze()
        high_contrast_noise_free_np = test_batch_high_contrast_noise_free.cpu().numpy().squeeze()
        low_contrast_noise_free_np = test_batch_low_contrast_noise_free.cpu().numpy().squeeze()

        # Evaluate PSNR
        psnr_value = psnr_metric(denoised_np, low_contrast_noise_free_np)
        print(f"PSNR: {psnr_value}")

        # Evaluate SSIM
        ssim_value = ssim_metric(denoised_np, low_contrast_noise_free_np)
        print(f"SSIM: {ssim_value}")

        # Evaluate VIF
        vif_value = vif_metric(high_contrast_noise_free_np, enhanced_np)
        print(f"VIF: {vif_value}")