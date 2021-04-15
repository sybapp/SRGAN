import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator

if __name__ == '__main__':

    UPSCALE_FACTOR = 4
    TEST_MODE = True if torch.cuda.is_available() else False
    IMAGE_NAME = "test.png"
    MODEL_NAME = "G_epoch_4_100.pth"

    model = Generator(UPSCALE_FACTOR).eval()
    if TEST_MODE:
        model.cuda()
        model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
    else:
        model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))

    image = Image.open(IMAGE_NAME)
    with torch.no_grad():
        image = Variable(ToTensor()(image)).unsqueeze(0)
    if TEST_MODE:
        image = image.cuda()

    start = time.process_time()
    out = model(image)
    elapsed = (time.process_time() - start)
    print('cost' + str(elapsed) + 's')
    out_img = ToPILImage()(out[0].data.cpu())
    out_img.save('out_srf_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)
