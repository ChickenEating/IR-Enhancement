import torch
import torch.nn as nn

# Multi-scale image decoupling
class MultiScaleDecoupling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleDecoupling, self).__init__()

        # Low-frequency (contrast component) branch
        self.low_freq_branch = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # High-frequency (detail component) branch
        self.high_freq_branch = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        low_freq = self.low_freq_branch(x)
        high_freq = self.high_freq_branch(x)
        return low_freq, high_freq

# Contrast Improvement Module (CIM) Network
class CIMNet(nn.Module):
    def __init__(self):
        super(CIMNet, self).__init__()
        self.multi_scale_decoupling = MultiScaleDecoupling(in_channels=1, out_channels=1)
        self.contrast_enhancer = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.detail_enhancer = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, denoised_image, detail_feature):
        low_freq, high_freq = self.multi_scale_decoupling(denoised_image)

        # Contrast enhancement
        contrast_enhanced = self.contrast_enhancer(low_freq)

        # Detail enhancement
        concatenated_detail = torch.cat(high_freq, detail_feature, dim=1)   # Concat detail information
        detail_enhanced = self.detail_enhancer(concatenated_detail)

        # Combine contrast and detail enhancements
        enhanced_image = denoised_image + contrast_enhanced + detail_enhanced

        return enhanced_image
