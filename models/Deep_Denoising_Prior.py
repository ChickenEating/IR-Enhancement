import os.path
import logging

import numpy as np
from collections import OrderedDict

import torch

from utils import utils_logger
from utils import utils_model
from utils import utils_image as image_utils


def main():
    input_noise_level = 0                 # AWGN noise level for input image
    model_noise_level = 5                 # Noise level for the model
    denoiser_model_name = 'drunet_gray'   # Denoiser model, 'drunet_gray' | 'drunet_color'
    dataset_name = 'L'                    # Test set,  'bsd68' | 'cbsd68' | 'set12'
    use_x8_boost = False                  # Default: False, x8 to boost performance
    display_image = False                  # Default: False
    border_size = 0                       # Shave border to calculate PSNR and SSIM

    if 'color' in denoiser_model_name:
        num_channels = 3                  # 3 for color image
    else:
        num_channels = 1                  # 1 for grayscale image

    model_directory = 'model_zoo'         # Fixed
    testset_directory = 'testsets'         # Fixed
    result_directory = 'results'          # Fixed
    task_type = 'dn'                      # 'dn' for denoising
    result_folder_name = dataset_name + '_' + task_type + '_' + denoiser_model_name

    model_file_path = os.path.join(model_directory, denoiser_model_name+'.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    
    # Input, Output, and Ground Truth Paths
    input_image_path = os.path.join(testset_directory, dataset_name)  # Input image path
    output_image_path = os.path.join(result_directory, result_folder_name)  # Output image path
    image_utils.create_directory(output_image_path)

    logger_name = result_folder_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(output_image_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    
    # Load Model
    from models.network_unet import UNetRes as DenoiserModel
    model = DenoiserModel(in_nc=num_channels+1, out_nc=num_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
    model.load_state_dict(torch.load(model_file_path), strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)
    logger.info('Model path: {:s}'.format(model_file_path))
    num_model_params = sum(p.numel() for p in model.parameters())
    logger.info('Number of parameters: {}'.format(num_model_params))

    evaluation_results = OrderedDict()
    evaluation_results['psnr'] = []
    evaluation_results['ssim'] = []

    logger.info('Model: {}, Model noise level: {}, Input noise level: {}'.format(denoiser_model_name, input_noise_level, model_noise_level))
    logger.info(input_image_path)
    input_image_files = image_utils.get_image_paths(input_image_path)

    for idx, image_file in enumerate(input_image_files):
        # (1) Load and Preprocess Input Image
        image_name, file_extension = os.path.splitext(os.path.basename(image_file))
        ground_truth_image = image_utils.read_image_uint(image_file, num_channels=num_channels)
        noisy_image = image_utils.uint_to_single(ground_truth_image)

        # Add noise without clipping
        np.random.seed(seed=0)  # for reproducibility
        noisy_image += np.random.normal(0, input_noise_level/255., noisy_image.shape)

        image_utils.display_image(image_utils.single_to_uint(noisy_image), title='Noisy image with noise level {}'.format(input_noise_level)) if display_image else None

        noisy_image_tensor = image_utils.single_to_tensor4(noisy_image)
        noisy_image_tensor = torch.cat((noisy_image_tensor, torch.FloatTensor([model_noise_level/255.]).repeat(1, 1, noisy_image_tensor.shape[2], noisy_image_tensor.shape[3])), dim=1)
        noisy_image_tensor = noisy_image_tensor.to(device)

        # (2) Denoise Image
        if not use_x8_boost and noisy_image_tensor.size(2)//8==0 and noisy_image_tensor.size(3)//8==0:
            denoised_image_tensor = model(noisy_image_tensor)
        elif not use_x8_boost and (noisy_image_tensor.size(2)//8!=0 or noisy_image_tensor.size(3)//8!=0):
            denoised_image_tensor = utils_model.test_mode(model, noisy_image_tensor, refield=64, mode=5)
        elif use_x8_boost:
            denoised_image_tensor = utils_model.test_mode(model, noisy_image_tensor, mode=3)

        denoised_image = image_utils.tensor_to_uint(denoised_image_tensor)

        # Calculate PSNR and SSIM
        if num_channels == 1:
            ground_truth_image = ground_truth_image.squeeze() 
        psnr_value = image_utils.calculate_psnr(denoised_image, ground_truth_image, border=border_size)
        ssim_value = image_utils.calculate_ssim(denoised_image, ground_truth_image, border=border_size)
        evaluation_results['psnr'].append(psnr_value)
        evaluation_results['ssim'].append(ssim_value)
        logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(image_name+file_extension, psnr_value, ssim_value))

        # Save Denoised Image
        image_utils.save_image(denoised_image, os.path.join(output_image_path, image_name+file_extension))

    average_psnr = sum(evaluation_results['psnr']) / len(evaluation_results['psnr'])
    average_ssim = sum(evaluation_results['ssim']) / len(evaluation_results['ssim'])
    logger.info('Average PSNR/SSIM(RGB) - {} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_folder_name, average_psnr, average_ssim))


if __name__ == '__main__':

    main()