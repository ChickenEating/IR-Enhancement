import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
from models.DDP import DDPNet
from models.CIM import CIMNet
from utils.Decomposition import decomposition
from utils.LossFunctions import calculate_ms_ssim_loss, calculate_adversarial_loss, calculate_contrast_loss, calculate_detail_loss

# Define our dataset class
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

# Initialize your dataset
dataset = IREDataset(root_dir="./data/IRE")

# Split dataset into training and validation sets
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders for training and validation
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize your DDPNet and CIMNet models
ddp_model = DDPNet()
cim_model = CIMNet()

# Choose your loss function and optimizer
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(list(ddp_model.parameters()) + list(cim_model.parameters()), lr=0.001)

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the models to the device
ddp_model.to(device)
cim_model.to(device)

# Training loop with validation
num_epochs = 10
for epoch in range(num_epochs):
    # Train loop
    ddp_model.train()
    cim_model.train()

    for batch_high_contrast_noisy, batch_high_contrast_noise_free, batch_low_contrast_noisy, batch_low_contrast_noise_free in train_loader:
        batch_high_contrast_noisy = batch_high_contrast_noisy.to(device)
        batch_high_contrast_noise_free = batch_high_contrast_noise_free.to(device)
        batch_low_contrast_noisy = batch_low_contrast_noisy.to(device)
        batch_low_contrast_noise_free = batch_low_contrast_noise_free.to(device)

        # DDPNet forward pass
        denoised_image, discriminator_output, detail_feautre = ddp_model(batch_low_contrast_noisy)

        # CIMNet forward pass
        enhanced_image = cim_model(denoised_image, detail_feautre)

        # loss calculation
        ms_ssim_loss = calculate_ms_ssim_loss(denoised_image, batch_low_contrast_noise_free)
        gan_loss = calculate_adversarial_loss(discriminator_output)
        contrast_component, detail_component = decomposition(enhanced_image)
        contrast_loss = calculate_contrast_loss(contrast_component, batch_high_contrast_noise_free)
        detail_loss = calculate_detail_loss(detail_component, batch_high_contrast_noise_free)

        # Total loss
        total_loss = gan_loss + 0.004 * ms_ssim_loss + 0.01 * detail_loss + 0.007 * contrast_loss

        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # Validation loop
    ddp_model.eval()
    cim_model.eval()
    with torch.no_grad():
        total_val_loss = 0.0
        for val_batch_high_contrast_noisy, val_batch_high_contrast_noise_free, val_batch_low_contrast_noisy, val_batch_low_contrast_noise_free in val_loader:
            val_batch_high_contrast_noisy = val_batch_high_contrast_noisy.to(device)
            val_batch_high_contrast_noise_free = val_batch_high_contrast_noise_free.to(device)
            val_batch_low_contrast_noisy = val_batch_low_contrast_noisy.to(device)
            val_batch_low_contrast_noise_free = val_batch_low_contrast_noise_free.to(device)

            # DDPNet forward pass
            val_denoised_image, _, val_detail_feautre = ddp_model(val_batch_low_contrast_noisy)

            # CIMNet forward pass
            val_enhanced_image = cim_model(val_denoised_image, val_detail_feautre)

            # loss calculation
            val_ms_ssim_loss = calculate_ms_ssim_loss(val_denoised_image, val_batch_low_contrast_noise_free)
            val_contrast_component, val_detail_component = decomposition(val_enhanced_image)
            val_contrast_loss = calculate_contrast_loss(val_contrast_component, val_batch_high_contrast_noise_free)
            val_detail_loss = calculate_detail_loss(val_detail_component, val_batch_high_contrast_noise_free)

            # Total validation loss
            total_val_loss += val_contrast_loss + 0.01 * val_detail_loss + 0.004 * val_ms_ssim_loss

    average_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_loss.item()}, Val Loss: {average_val_loss.item()}")

# Save the trained models if needed
torch.save(ddp_model.state_dict(), "ddp_model.pth")
torch.save(cim_model.state_dict(), "cim_model.pth")
