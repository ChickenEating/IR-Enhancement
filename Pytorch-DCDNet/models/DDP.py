import torch.nn as nn
from DAU import DAU

# U-Net architecture
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.dau = DAU(in_channels, out_channels)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        x = self.encoder(x)
        detail_feature = self.dau(x)    # Detail information extraction
        x = self.decoder(x)
        return x, detail_feature

# Deep Denoising Prior (DDP) Module
class DDPNet(nn.Module):
    def __init__(self):
        super(DDPNet, self).__init__()
        self.generator = UNet(in_channels=1, out_channels=1)  # Adjust channels based on your input and output requirements
        self.discriminator = UNet(in_channels=1, out_channels=1)  # You can customize the discriminator architecture

    def forward(self, input_image):
        denoised_image, detail_feature = self.generator(input_image)
        discriminator_output = self.discriminator(denoised_image)
        return denoised_image, discriminator_output, detail_feature
