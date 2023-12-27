import torch
import torch.nn as nn

class DAU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DAU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Subtract the central pixel value from each pixel in the 3x3 convolution output
        center_pixel_value = self.conv(x)[:, :, 1, 1].unsqueeze(2).unsqueeze(3)
        detail_feature = self.conv(x) - center_pixel_value
        return detail_feature
    