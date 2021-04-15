import os
from math import log10

import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TestDatasetFromFolder, display_transform
from model import Generator_OSRGAN

if __name__ == '__main__':

    UPSCALE_FACTOR = 4
    MODEL_NAME = 'G_OSRGAN_epoch_4_100.pth'

    results = {'Set5': {'psnr': [], 'ssim': []},
               'Set14': {'psnr': [], 'ssim': []},
               'BSD100': {'psnr': [], 'ssim': []},
               'Urban100': {'psnr': [], 'ssim': []}
               }

    model = Generator_OSRGAN(UPSCALE_FACTOR).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

    test_set = TestDatasetFromFolder('data/test', upscale_factor=UPSCALE_FACTOR)
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
    test_bar = tqdm(test_loader, desc='[testing datasets]')

    out_path = 'test_results/SRF_OSRGAN_' + str(UPSCALE_FACTOR) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for image_name, lr_image, hr_restore_img, hr_image in test_bar:
        image_name = image_name[0]
        with torch.no_grad():
            lr_image = Variable(lr_image)
            hr_image = Variable(hr_image)
        if torch.cuda.is_available():
            lr_image = lr_image.cuda()
            hr_image = hr_image.cuda()
        with torch.no_grad():
            sr_image = model(lr_image)
        mse = ((hr_image - sr_image) ** 2).data.mean()
        psnr = 10 * log10(1 / mse)
        ssim = pytorch_ssim.ssim(sr_image, hr_image).data

        test_images = torch.stack(
            [display_transform()(hr_restore_img.squeeze(0)), display_transform()(hr_image.data.cpu().squeeze(0)),
             display_transform()(sr_image.data.cpu().squeeze(0))])
        image = utils.make_grid(test_images, nrow=3, padding=5)
        utils.save_image(image, out_path + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) +
                         image_name.split('.')[-1], padding=5)

        # 保存 psnr\ssim
        results[image_name.split('_')[0]]['psnr'].append(psnr)
        results[image_name.split('_')[0]]['ssim'].append(ssim)

    out_path = 'statistics/'
    saved_results = {'psnr': [], 'ssim': []}
    for item in results.values():
        psnr = np.array(item['psnr'])
        ssim = np.array([i.item() for i in item['ssim']])
        if (len(psnr) == 0) or (len(ssim) == 0):
            psnr = 'No data'
            ssim = 'No data'
        else:
            psnr = psnr.mean()
            ssim = ssim.mean()
        saved_results['psnr'].append(psnr)
        saved_results['ssim'].append(ssim)

    data_frame = pd.DataFrame(saved_results, results.keys())
    data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_test_results_osrgan.csv', index_label='DataSet')
