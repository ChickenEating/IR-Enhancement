import numpy as np
import cv2
import scipy


def MSE(img1, img2):
    #判断两张图片的shape是否一致
    assert img1.shape == img2.shape,"images have different shape " \
                                    "{} and {}".format(img1.shape,img2.shape)
    #计算两张图像的MSE
    img_mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64))**2)
    #将图像的MSE量化为相似度
    img_sim = img_mse / (255**2)
    return img_mse,1 - img_sim

def PSNR(img1, img2):
    #判断两张图片的size是否一致
    assert img1.shape == img2.shape,"images hava different shape " \
                                    "{} and {}".format(img1.shape,img2.shape)
    #获取数据类型所表示的最大值
    MAX = np.iinfo(img1.dtype).max
    #计算两张图片的MSE
    mse,_ = MSE(img1, img2)
    #计算两张图片的PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10(MAX**2 / mse)

    return psnr

def SSIM(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
    
def VIF(ref, dist):
    sigma_nsq = 2
    eps = 1e-10
    num = 0.0
    den = 0.0
    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0
        if scale > 1:
            ref = scipy.ndimage.gaussian_filter(ref, sd)
            dist = scipy.ndimage.gaussian_filter(dist, sd)
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]
        mu1 = scipy.ndimage.gaussian_filter(ref, sd)
        mu2 = scipy.ndimage.gaussian_filter(dist, sd)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = scipy.ndimage.gaussian_filter(ref * ref, sd) - mu1_sq
        sigma2_sq = scipy.ndimage.gaussian_filter(dist * dist, sd) - mu2_sq
        sigma12 = scipy.ndimage.gaussian_filter(ref * dist, sd) - mu1_mu2
        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0
        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12
        g[sigma1_sq < eps] = 0
        sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
        sigma1_sq[sigma1_sq < eps] = 0
        g[sigma2_sq < eps] = 0
        sv_sq[sigma2_sq < eps] = 0
        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= eps] = eps
        num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))
    vifp = num / den
    if np.isnan(vifp):
        return 1.0
    else:
        return vifp